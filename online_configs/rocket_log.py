online_dict = {
    "trainer_name": "PPOTrainer",
    "detach_rollout_manager": True,
    "rollout_config": {
        "num_rollout_workers": 2,  #! 3
        "num_gpus_per_worker": 1.0,
        "num_cpus_per_worker": 1,
        "fragment_length": 128,  #! 256
        "to_send_queue_size": 8,
        "worker_config": {
            "num_envs": 16,  #! 16
            "batch_size": 8,  #! 8
            "restart_interval": 3600,  # 1h
            "video_fps": 20,
            "video_output_dir": "output/videos",
        },
        "replay_buffer_config": {
            "max_chunks": 4800,
            "max_reuse": 2,
            "max_staleness": 2,
            "fragments_per_report": 40,
            "fragments_per_chunk": 1,
            "database_config": {
                "path": "output/replay_buffer_cache",
                "num_shards": 8,
            },
        },
        "episode_statistics_config": {},
    },
    "train_config": {
        "num_workers": 6,  #! 4
        "num_gpus_per_worker": 1.0,
        "num_iterations": 4000,
        "vf_warmup": 0,
        "learning_rate": 0.00002,
        "anneal_lr_linearly": False,
        "weight_decay": 0.04,
        "adam_eps": 1e-8,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 100,  #! 200
        "gradient_accumulation": 5,  #! 10  # TODO: check
        "epochs_per_iteration": 1,  # TODO: check
        "context_length": 64,  #! 64
        "discount": 0.999,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "clip_vloss": False,  # TODO: check
        "max_grad_norm": 5,  # ????
        "zero_initial_vf": True,
        "ppo_policy_coef": 1.0,
        "ppo_vf_coef": 0.5,  # TODO: check
        "kl_divergence_coef_rho": 0.2,
        "entropy_bonus_coef": 0.0,
        "coef_rho_decay": 0.9995,
        "log_ratio_range": 50,  # for numerical stability
        "normalize_advantage_full_batch": True,  # TODO: check!!!
        "use_normalized_vf": True,
        "num_readers": 4,
        "num_cpus_per_reader": 0.1,
        "prefetch_batches": 4,  #! 2
        "save_interval": 10,
        "keep_interval": 40,
        "record_video_interval": 1,  #! 2
        "enable_ref_update": True,
        "resume": None,
        "resume_optimizer": True,
        "save_path": "output",
    },
    "logger_config": {"project": "minestudio_online", "name": "rocket_log"},
}


def env_generator():
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        CommandsCallback,
        FastResetCallback,
        JudgeResetCallback,
        PrevActionCallback,
        SummonMobsCallback,
    )

    from rocket_callbacks import (
        BlockConfig,
        ResetConfig,
        RocketOnlineCallback,
        SpawnBlocks,
    )

    blocks = [
        "diamond_block",
        "diamond_ore",
        "gold_ore",
        "coal_ore",
        "iron_ore",
        "redstone_ore",
        "crafting_table",
        "furnace",
        "anvil",
        "chest",
        "oak_planks",
        "torch",
        "sunflower",
        "poppy",
        "dandelion",
    ]

    re_pattern = f".*({'|'.join(blocks)}|log).*"

    env = MinecraftSim(
        obs_size=(224, 224),
        preferred_spawn_biome="plains",
        num_empty_frames=100,
        callbacks=[
            SummonMobsCallback(
                [
                    {
                        "name": "sheep",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "cow",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "pig",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "llama",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "chicken",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "cat",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "horse",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "wolf",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                    {
                        "name": "rabbit",
                        "number": 1,
                        "range_x": [-10, 10],
                        "range_z": [-5, 15],
                    },
                ]
            ),
            CommandsCallback(
                commands=[
                    "/gamerule sendCommandFeedback false",
                    "/give @p minecraft:diamond_pickaxe 1",
                ]
            ),
            SpawnBlocks(
                block_configs=[
                    BlockConfig(
                        names=blocks,
                        options=6,
                        number=2,
                        range_x=[-15, 15],
                        range_y=[0, 1],
                        range_z=[-15, 15],
                    )
                ]
            ),
            FastResetCallback(
                biomes=["forest", "plains", "desert", "savanna", "mountains"],
                random_tp_range=1000,
            ),
            JudgeResetCallback(500),
            RocketOnlineCallback(
                reset_configs=[
                    ResetConfig(
                        views=[
                            (15, 10, 0, 45, 45),  # p1, left
                            (-15, 10, 0, -45, 45),  # p2, right
                            (0, 10, -10, 0, 45),  # p3, back
                            (10, 10, -10, 30, 30),  # p4, back - left
                            (-10, 10, -10, -30, 30),  # p5, back - right
                            (10, 2, -10, 30, 15),  # p4, back - left
                            (-10, 2, -10, -30, 15),  # p5, back - right
                            (0, 2, 0, 0, 0),  # p6, back - right
                        ],
                        voxel_range=[-40, 40, -15, 5, -40, 40],
                        re_pattern=re_pattern,
                        obj_id=2,
                    ),
                ]
            ),
            PrevActionCallback(),
        ],
    )
    return env


def policy_generator(checkpoint_path: str):
    from model import load_cross_view_rocket

    policy = load_cross_view_rocket(checkpoint_path).to("cuda")
    return policy
