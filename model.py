"""ROCKET-3 cross-view visuomotor policy."""

from typing import Dict, List, Optional

import timm
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from minestudio.models.base_policy import MinePolicy
from minestudio.online.utils import auto_to_torch
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from rich import print
from torch import nn

BINARY_KEYS = [
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


class ActionEmbeddingLayer(nn.Module):

    def __init__(self, hiddim: int):
        super().__init__()
        self.camera_layer = nn.Linear(2, hiddim)
        self.binary_layers = nn.ModuleDict(
            {f"act_{key}": nn.Embedding(2, hiddim) for key in BINARY_KEYS}
        )

    def forward(self, action: Dict) -> torch.Tensor:
        x = self.camera_layer(action["camera"].float())
        for key in BINARY_KEYS:
            x += self.binary_layers[f"act_{key}"](action[key.replace("_", ".")])
        return x


class CrossViewRocket(MinePolicy, PyTorchModelHubMixin):

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
        # super().__init__(hiddim=hiddim, action_space=action_space, nucleus_prob=0.85)
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
        self.interaction = nn.Embedding(
            10, hiddim
        )  # denotes the number of interaction types
        self.num_step_tokens = self.num_view_tokens + 1

        self.index_bias = -2
        self.use_prev_action = use_prev_action
        if self.use_prev_action:
            self.action_embedding_layer = ActionEmbeddingLayer(hiddim)
            self.num_step_tokens += 1
            self.index_bias -= 1

        self.dropout_embedding = nn.Parameter(torch.randn(1, 1, hiddim) * 1e-3)

        print(f"number of tokens per timestep is {self.num_step_tokens}")
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

        self.aux_vis_head = nn.Linear(hiddim, 1 + 2 + 4)  # exist, point, bbox
        # self.pre_aux_vis_head = nn.Linear(hiddim, 1+2+4) # exist, point, bbox

        #! view backbone is frozen during training
        for param in self.view_backbone.parameters():
            param.requires_grad = False

    def encode_view_tokens(
        self, agent_view: torch.Tensor, cross_view: Dict
    ) -> torch.Tensor:
        b, t = agent_view.shape[:2]

        # 1. encode observation with view_backbone
        obs_rgb = rearrange(agent_view, "b t h w c -> (b t) c h w")
        obs_rgb = self.transforms(obs_rgb)
        x_obs = self.view_backbone(obs_rgb)[-1]
        x_obs = self.updim_obs(x_obs)
        x_obs = rearrange(x_obs, "b c h w -> b (h w) c")

        # 2. encode cross-view image with view_backbone
        cross_view_rgb = rearrange(
            cross_view["cross_view_image"], "b t h w c -> (b t) c h w"
        )
        cross_view_rgb = self.transforms(cross_view_rgb)
        x_cross_image = self.view_backbone(cross_view_rgb)[-1]

        # 3. encode cross-view object mask with mask_backbone
        cross_view_mask = cross_view["cross_view_obj_mask"]
        cross_view_mask = rearrange(cross_view_mask, "b t h w -> (b t) 1 h w") * 1.0
        x_cross_mask = self.mask_backbone(cross_view_mask)[-1]

        # 4. fuse x_cross image and x_cross mask in the feature dimension
        x_cross = torch.cat([x_cross_image, x_cross_mask], dim=1)
        x_cross = self.updim_cross(x_cross)
        x_cross = rearrange(x_cross, "b c h w -> b (h w) c")

        # 5. fuse x_obs and x_cross in the spatial dimension
        # generate view tokens
        x_cls = self.view_cls_tokens.expand(x_obs.shape[0], -1, -1)
        x_view = torch.cat([x_cls, x_obs, x_cross], dim=1)
        x_view = self.view_resampler(x_view)[:, : self.num_view_tokens, :]
        x_view = rearrange(x_view, "(b t) n c -> b t n c", b=b)

        return x_view

    def temporal_reason(
        self, x: torch.Tensor, memory: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        b, t = x.shape[:2]
        if not hasattr(self, "first") or self.first.shape[:2] != (b, t):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        z, memory = self.recurrent(x, self.first, memory)
        z = F.relu(z, inplace=False)
        z = self.lastlayer(z)
        z = self.final_ln(z)
        return z, memory

    def forward(
        self, input: Dict, state_in: Optional[List[torch.Tensor]] = None, **kwargs
    ) -> Dict:

        b, t = input["image"].shape[:2]

        x_view = self.encode_view_tokens(input["image"], input["cross_view"])
        x = x_view

        # generate interaction tokens
        x_cond = self.interaction(input["cross_view"]["cross_view_obj_id"] + 1)
        x_cond = rearrange(x_cond, "b t c -> b t 1 c")
        x = torch.cat([x, x_cond], dim=-2)

        # generate prev_action tokens
        if self.use_prev_action:
            x_prev_a = self.action_embedding_layer(input["env_prev_action"])
            if "prev_action_dropout" in input:
                dropout_mask = input["prev_action_dropout"][..., None]
                dropout_embedding = repeat(
                    self.dropout_embedding, "1 1 c -> b t c", b=b, t=t
                )
                x_prev_a = x_prev_a * dropout_mask + dropout_embedding * (
                    1 - dropout_mask
                )

            # dropout_embedding = repeat(self.dropout_embedding, "1 1 c -> b t c", b=b, t=t) #! only for debug
            # x_prev_a = dropout_embedding #! only for debug

            x_prev_a = rearrange(x_prev_a, "b t c -> b t 1 c")
            x = torch.cat([x, x_prev_a], dim=-2)

        x = rearrange(x, "b t n c -> b (t n) c", b=b)
        z, state_out = self.temporal_reason(x, state_in)
        z = rearrange(z, "b (t n) c -> b t n c", t=t)

        aux_vis_logits = self.aux_vis_head(z[:, :, self.index_bias, :])
        exist = aux_vis_logits[:, :, 0:1]
        point = aux_vis_logits[:, :, 1:3]
        bbox = aux_vis_logits[:, :, 3:7]

        pi_logits = self.pi_head(z[:, :, -1, :])
        vpred = self.value_head(z[:, :, -1, :])
        latents = {
            "pi_logits": pi_logits,
            "vpred": vpred,
            "exist": exist,
            "point": point,
            "bbox": bbox,
        }
        return latents, state_out

    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        if batch_size is None:
            return [
                t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)
            ]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

    # def merge_input(self, inputs) -> torch.tensor:
    #     print(f"{inputs[0].keys() = }; {inputs[0]['image'].shape = }")
    #     inputs = auto_to_torch(inputs, device=self.device)
    #     out_inputs = auto_to_torch(auto_stack([auto_stack([input]) for input in inputs]), device=self.device)
    #     return out_inputs
    #     if inputs[0]["image"].dim() == 3:
    #         # in_inputs=[{"image": input["image"]} for input in inputs]
    #         out_inputs = auto_to_torch(auto_stack([auto_stack([input]) for input in in_inputs]), device=self.device)
    #         return out_inputs
    #     elif inputs[0]["image"].dim() == 4:
    #         out_inputs = auto_to_torch(auto_stack([input["image"] for input in inputs]), device=self.device)
    #         return out_inputs

    def merge_state(self, states) -> Optional[List[torch.Tensor]]:
        result_states = []
        for i in range(len(states[0])):
            result_states.append(
                auto_to_torch(
                    torch.cat([state[i] for state in states], 0), device=self.device
                )
            )
        return result_states

    def split_state(self, states, split_num) -> Optional[List[List[torch.Tensor]]]:
        result_states = [
            [states[j][i : i + 1] for j in range(len(states))] for i in range(split_num)
        ]
        return result_states


def load_cross_view_rocket(
    ckpt_path: str, model_config: Optional[Dict] = None
) -> CrossViewRocket:
    """Load a full training checkpoint or a ROCKET-3 state dictionary."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        config = dict(checkpoint["hyper_parameters"]["model"])
        weights = checkpoint["state_dict"]
    else:
        weights = checkpoint
        config = {}

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
    config["pretrained_backbone"] = False
    model = CrossViewRocket(**config)
    model.load_state_dict(state_dict, strict=True)
    return model
