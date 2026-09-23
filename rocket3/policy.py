"""ROCKET-3 policy for the cross-view goal in paper Sec. 4 and Appendix D.

The vision and temporal backbone starts from the ROCKET-2 imitation-learning
policy; ROCKET-3 post-training optimizes its action policy with RL. This module
defines the policy, not the PPO objective implemented by MineStudio.
"""

import logging

import timm
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from minestudio.models.base_policy import MinePolicy
from minestudio.online.utils import auto_to_torch
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from torch import nn

logger = logging.getLogger(__name__)

BINARY_ACTION_KEYS = [
    "forward",
    "back",
    "left",
    "right",
    "inventory",
    "sprint",
    "sneak",
    "jump",
    "attack",
    "use",
    "hotbar_1",
    "hotbar_2",
    "hotbar_3",
    "hotbar_4",
    "hotbar_5",
    "hotbar_6",
    "hotbar_7",
    "hotbar_8",
    "hotbar_9",
]


class PreviousActionEmbedding(nn.Module):
    """Embed the previous Minecraft action as one optional temporal token."""

    def __init__(self, hiddim: int):
        super().__init__()
        self.camera_layer = nn.Linear(2, hiddim)
        self.binary_layers = nn.ModuleDict(
            {f"act_{key}": nn.Embedding(2, hiddim) for key in BINARY_ACTION_KEYS}
        )

    def forward(self, action: dict) -> torch.Tensor:
        # MineStudio spells hotbar keys with dots in observations (hotbar.1).
        action_features = self.camera_layer(action["camera"].float())
        for key in BINARY_ACTION_KEYS:
            action_features += self.binary_layers[f"act_{key}"](
                action[key.replace("_", ".")]
            )
        return action_features


class RocketPolicy(MinePolicy, PyTorchModelHubMixin):
    """Condition actions on the current view and a masked goal view.

    Paper notation: ``image`` is O_t; the cross-view image/mask are O_g/M_g;
    ``cross_view_obj_id`` encodes interaction event E. The output includes the
    action policy plus the visibility and target-location auxiliary predictions
    used during imitation learning (paper Eq. 4 and Appendix D).
    """

    def __init__(
        self,
        view_backbone: str = "timm/vit_base_patch16_224.dino",
        mask_backbone: str = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        hiddim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        timesteps: int = 128,
        mem_len: int = 128,
        use_prev_action: bool = False,
        num_view_tokens: int = 1,
        action_space=None,
        pretrained_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space)
        # The DINO RGB encoder is frozen, while the mask encoder is trainable
        # (paper Appendix D). A checkpoint already contains backbone weights.
        self.view_backbone = timm.create_model(
            view_backbone, pretrained=pretrained_backbone, features_only=True
        )
        data_config = timm.data.resolve_model_data_config(self.view_backbone)
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: x / 255.0),
                torchvision.transforms.Normalize(
                    mean=data_config["mean"], std=data_config["std"]
                ),
            ]
        )
        self.mask_backbone = timm.create_model(
            mask_backbone,
            pretrained=pretrained_backbone,
            features_only=True,
            in_chans=1,
        )
        self.updim_obs = nn.Conv2d(
            self.view_backbone.feature_info[-1]["num_chs"],
            hiddim,
            kernel_size=1,
            bias=False,
        )
        vision_dim = (
            self.view_backbone.feature_info[-1]["num_chs"]
            + self.mask_backbone.feature_info[-1]["num_chs"]
        )
        self.updim_cross = nn.Conv2d(vision_dim, hiddim, kernel_size=1, bias=False)
        self.num_view_tokens = num_view_tokens
        # Learned queries compress the current/goal patch tokens into a fixed
        # number of view tokens before temporal reasoning.
        self.view_cls_tokens = nn.Parameter(
            torch.randn(1, self.num_view_tokens, hiddim) * 1e-3
        )
        self.view_resampler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim * 2,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        # Add one before lookup so -1 represents an absent goal/event.
        self.interaction = nn.Embedding(10, hiddim)
        self.num_step_tokens = self.num_view_tokens + 1

        # The auxiliary heads read the last view token. The action/value heads
        # read the last token of each step, after the event/optional action.
        self.aux_view_token_index = -2
        self.use_prev_action = use_prev_action
        if self.use_prev_action:
            self.action_embedding_layer = PreviousActionEmbedding(hiddim)
            self.num_step_tokens += 1
            self.aux_view_token_index -= 1

        self.dropout_embedding = nn.Parameter(torch.randn(1, 1, hiddim) * 1e-3)

        logger.debug("Tokens per timestep: %s", self.num_step_tokens)
        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps * self.num_step_tokens,
            recurrence_type="transformer",
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4,
            pointwise_use_activation=False,
            attention_mask_style="clipped_causal",
            attention_heads=num_heads,
            attention_memory_size=(mem_len + timesteps) * self.num_step_tokens,
            n_block=num_layers,
            inject_condition=False,  # inject obj_embedding as the condition
        )
        self.lastlayer = FanInInitReLULayer(
            hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True
        )
        self.final_ln = nn.LayerNorm(hiddim)

        # Visibility logit, 2-D centroid and 4-D box. The paper reports that
        # visibility/centroid remain useful after RL without auxiliary labels.
        self.aux_vis_head = nn.Linear(hiddim, 1 + 2 + 4)

        for param in self.view_backbone.parameters():
            param.requires_grad = False

    def encode_view_tokens(
        self, agent_view: torch.Tensor, cross_view: dict
    ) -> torch.Tensor:
        """Fuse RGB O_t, RGB O_g and binary M_g into ``(B, T, N, D)`` tokens.

        The two ViT-B/16 RGB feature maps and ViT-tiny/16 mask feature map
        each have a 14 x 14 patch grid for 224 x 224 input (Appendix D).
        """
        b, t = agent_view.shape[:2]

        # Flatten batch/time for the shared, frozen RGB encoder.
        agent_rgb = rearrange(agent_view, "b t h w c -> (b t) c h w")
        agent_rgb = self.transforms(agent_rgb)
        agent_patches = self.view_backbone(agent_rgb)[-1]
        agent_patches = self.updim_obs(agent_patches)
        agent_patches = rearrange(agent_patches, "b c h w -> b (h w) c")

        # The same RGB encoder processes the goal image.
        goal_rgb = rearrange(cross_view["cross_view_image"], "b t h w c -> (b t) c h w")
        goal_rgb = self.transforms(goal_rgb)
        goal_rgb_patches = self.view_backbone(goal_rgb)[-1]

        # The trainable mask encoder receives a one-channel 0/1 target mask.
        goal_mask = cross_view["cross_view_obj_mask"]
        goal_mask = rearrange(goal_mask, "b t h w -> (b t) 1 h w") * 1.0
        goal_mask_patches = self.mask_backbone(goal_mask)[-1]

        # Align goal RGB and mask patches by position, then fuse channels.
        goal_patches = torch.cat([goal_rgb_patches, goal_mask_patches], dim=1)
        goal_patches = self.updim_cross(goal_patches)
        goal_patches = rearrange(goal_patches, "b c h w -> b (h w) c")

        # Non-causal spatial attention aligns the current and goal views.
        # Learned query tokens retain the condensed per-frame representation.
        query_tokens = self.view_cls_tokens.expand(agent_patches.shape[0], -1, -1)
        spatial_tokens = torch.cat([query_tokens, agent_patches, goal_patches], dim=1)
        view_tokens = self.view_resampler(spatial_tokens)[:, : self.num_view_tokens, :]
        return rearrange(view_tokens, "(b t) n c -> b t n c", b=b)

    def temporal_reason(
        self, tokens: torch.Tensor, memory: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply causal Transformer-XL reasoning over flattened step tokens.

        ``memory`` carries the prior K-V state across fragments, matching the
        long-history design discussed in paper Sec. 4 and Appendix B.2.
        """
        b, t = tokens.shape[:2]
        if not hasattr(self, "first") or self.first.shape[:2] != (b, t):
            self.first = torch.tensor([[False]], device=tokens.device).repeat(b, t)
        if memory is None:
            memory = [
                state.to(tokens.device) for state in self.recurrent.initial_state(b)
            ]
        temporal_features, memory = self.recurrent(tokens, self.first, memory)
        temporal_features = F.relu(temporal_features, inplace=False)
        temporal_features = self.lastlayer(temporal_features)
        temporal_features = self.final_ln(temporal_features)
        return temporal_features, memory

    def forward(
        self, input: dict, state_in: list[torch.Tensor] | None = None, **kwargs
    ) -> tuple[dict, list[torch.Tensor]]:
        """Return policy/value and auxiliary outputs, plus recurrent state.

        MineStudio's trainer calls this method with the keyword ``input``.
        """

        b, t = input["image"].shape[:2]

        view_tokens = self.encode_view_tokens(input["image"], input["cross_view"])

        # The interaction ID is constant for a goal; -1 maps to the null slot.
        event_token = self.interaction(input["cross_view"]["cross_view_obj_id"] + 1)
        event_token = rearrange(event_token, "b t c -> b t 1 c")
        step_tokens = torch.cat([view_tokens, event_token], dim=-2)

        # Optional previous-action conditioning follows the loaded checkpoint.
        if self.use_prev_action:
            previous_action_token = self.action_embedding_layer(
                input["env_prev_action"]
            )
            if "prev_action_dropout" in input:
                # Imitation data may hide this token so the policy cannot rely
                # solely on the previous control instead of visual evidence.
                dropout_mask = input["prev_action_dropout"][..., None]
                dropout_embedding = repeat(
                    self.dropout_embedding, "1 1 c -> b t c", b=b, t=t
                )
                previous_action_token = (
                    previous_action_token * dropout_mask
                    + dropout_embedding * (1 - dropout_mask)
                )

            previous_action_token = rearrange(previous_action_token, "b t c -> b t 1 c")
            step_tokens = torch.cat([step_tokens, previous_action_token], dim=-2)

        temporal_input = rearrange(step_tokens, "b t n c -> b (t n) c", b=b)
        temporal_features, state_out = self.temporal_reason(temporal_input, state_in)
        temporal_features = rearrange(temporal_features, "b (t n) c -> b t n c", t=t)

        # Separate spatial probes from the final token used for control.
        aux_vis_logits = self.aux_vis_head(
            temporal_features[:, :, self.aux_view_token_index, :]
        )
        exist = aux_vis_logits[:, :, 0:1]
        point = aux_vis_logits[:, :, 1:3]
        bbox = aux_vis_logits[:, :, 3:7]

        control_features = temporal_features[:, :, -1, :]
        pi_logits = self.pi_head(control_features)
        vpred = self.value_head(control_features)
        latents = {
            "pi_logits": pi_logits,
            "vpred": vpred,
            "exist": exist,
            "point": point,
            "bbox": bbox,
        }
        return latents, state_out

    def initial_state(self, batch_size: int | None = None) -> list[torch.Tensor]:
        if batch_size is None:
            return [
                t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)
            ]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

    def merge_state(self, states) -> list[torch.Tensor]:
        """Batch per-environment K-V states for MineStudio rollout inference."""
        result_states = []
        for i in range(len(states[0])):
            result_states.append(
                auto_to_torch(
                    torch.cat([state[i] for state in states], 0), device=self.device
                )
            )
        return result_states

    def split_state(self, states, split_num) -> list[list[torch.Tensor]]:
        """Return one recurrent state per environment after batched inference."""
        result_states = [
            [states[j][i : i + 1] for j in range(len(states))] for i in range(split_num)
        ]
        return result_states


def load_rocket3_policy(
    ckpt_path: str, model_config: dict | None = None
) -> RocketPolicy:
    """Load either a MineStudio training checkpoint or bare ROCKET-3 weights.

    Bare weights carry no model config, so infer the embedding width, number of
    goal-view tokens and previous-action conditioning from tensor names/shapes.
    Other non-default architecture choices need ``model_config``.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        config = dict(checkpoint["hyper_parameters"]["model"])
        weights = checkpoint["state_dict"]
    else:
        weights = checkpoint
        config = {}

    # A trainer may nest the policy under ``mine_policy`` in its state dict.
    state_dict = {key.removeprefix("mine_policy."): val for key, val in weights.items()}
    if "view_cls_tokens" not in state_dict:
        raise ValueError("Checkpoint does not contain ROCKET-3 policy weights")
    if "state_dict" not in checkpoint:
        view_tokens = state_dict["view_cls_tokens"]
        config.update(
            hiddim=view_tokens.shape[-1],
            num_view_tokens=view_tokens.shape[1],
            use_prev_action=any(
                key.startswith("action_embedding_layer.") for key in state_dict
            ),
        )
    if model_config:
        config.update(model_config)
    # The checkpoint supplies both backbones; avoid fetching timm weights.
    config["pretrained_backbone"] = False
    model = RocketPolicy(**config)
    model.load_state_dict(state_dict, strict=True)
    return model


# Preserve existing imports and checkpoint module names while making the new
# package API describe each component's role.
BINARY_KEYS = BINARY_ACTION_KEYS
ActionEmbeddingLayer = PreviousActionEmbedding
CrossViewRocket = RocketPolicy
load_cross_view_rocket = load_rocket3_policy
