from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from drone_pufferlib.training.adapter import configure_pufferl_runtime, get_adapter_info, make_vector_env, require_pufferlib
from drone_pufferlib.training.common import build_model, build_puffer_train_config, load_train_bundle, resolve_device, save_checkpoint
from drone_pufferlib.training.evaluate import evaluate_policy, evaluate_random_policy
from drone_pufferlib.utils.seeding import set_global_seeds
from drone_pufferlib.utils.visualization import append_metrics_csv, plot_training_curves, save_json


def should_overwork(requested_workers: int) -> bool:
    import psutil

    physical_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or requested_workers
    return requested_workers > int(physical_cores)


def initialize_distributed(device_name: str) -> tuple[dict[str, int] | None, torch.device]:
    if "LOCAL_RANK" not in os.environ:
        return None, resolve_device(device_name)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(device_name)

    return {"local_rank": local_rank, "rank": rank, "world_size": world_size}, device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def synchronize_stop(should_stop: bool, device: torch.device) -> bool:
    if not is_distributed():
        return should_stop
    sync_device = device if device.type != "cpu" else torch.device("cpu")
    flag = torch.tensor(int(should_stop), device=sync_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def attach_ddp(model: torch.nn.Module, distributed_cfg: dict[str, int] | None) -> tuple[torch.nn.Module, torch.nn.Module]:
    if distributed_cfg is None:
        return model, model

    local_rank = distributed_cfg["local_rank"]
    if next(model.parameters()).device.type == "cuda":
        wrapped = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        wrapped = DDP(model)
    wrapped.forward_eval = model.forward_eval
    wrapped.hidden_size = model.hidden_size
    return model, wrapped


def effective_env_steps(local_steps: int) -> int:
    return local_steps * get_world_size()


def close_trainer(trainer, *, main_process: bool) -> None:
    if trainer is None:
        return
    if is_distributed() and not main_process:
        trainer.vecenv.close()
        trainer.utilization.stop()
        return
    trainer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted.path=value")
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    args = parser.parse_args()

    train_cfg, env_cfg = load_train_bundle(args.config, overrides=args.override)
    distributed_cfg, device = initialize_distributed(train_cfg["device"])
    rank = get_rank()
    world_size = get_world_size()
    main_process = is_main_process()

    set_global_seeds(int(train_cfg["seed"]))
    artifacts_dir = Path(train_cfg["artifacts_dir"])
    writer = SummaryWriter(log_dir=artifacts_dir / "logs" / train_cfg["run_name"]) if main_process else None
    adapter_info = get_adapter_info()
    require_pufferlib()
    pufferl = __import__("pufferlib.pufferl", fromlist=["PuffeRL"])
    configure_pufferl_runtime(pufferl)
    trainer = None

    base_model, policy = attach_ddp(build_model(env_cfg, train_cfg["model"], device), distributed_cfg)

    num_workers = int(train_cfg["vectorization"]["num_workers"])
    overwork = should_overwork(num_workers)
    if overwork and main_process:
        print(
            f"Enabling vector overwork for {num_workers} workers on this host "
            "to preserve the configured environment count."
        )

    envs_per_worker = int(train_cfg["vectorization"]["envs_per_worker"])
    num_envs = num_workers * envs_per_worker
    base_seed = int(train_cfg["env"]["train_seed_start"])
    seed_stride = int(train_cfg["env"]["train_seed_stride"])
    rank_seed_offset = rank * num_envs
    seeds = [base_seed + (rank_seed_offset + idx) * seed_stride for idx in range(num_envs)]
    rollout_length = int(train_cfg["training"]["rollout_length"])

    vector_env = make_vector_env(
        env_cfg,
        seeds=seeds,
        multiprocessing=bool(train_cfg["vectorization"]["multiprocessing"]),
        difficulty=train_cfg["env"]["difficulty"],
        num_workers=num_workers,
        batch_size=num_envs,
        overwork=overwork,
    )
    puffer_train_cfg = build_puffer_train_config(
        train_cfg,
        num_envs=num_envs,
        rollout_length=rollout_length,
        world_size=world_size,
    )
    trainer = pufferl.PuffeRL(puffer_train_cfg, vector_env, policy)
    total_env_steps = int(train_cfg["training"]["total_env_steps"])
    eval_interval = int(train_cfg["training"]["eval_interval"])
    next_eval = eval_interval
    eval_steps: list[int] = []
    eval_rewards: list[float] = []
    eval_successes: list[float] = []
    start_time = time.monotonic()

    if main_process:
        random_baseline = evaluate_random_policy(
            env_cfg,
            episodes=int(train_cfg["evaluation"]["random_policy_episodes"]),
            seed_start=int(train_cfg["env"]["eval_seed_start"]),
            difficulty="easy",
        )
        save_json(artifacts_dir / "eval" / "random_baseline.json", random_baseline)
    barrier()

    try:
        while effective_env_steps(int(trainer.global_step)) < total_env_steps:
            trainer.evaluate()
            logs = trainer.train() or {}
            env_steps = effective_env_steps(int(trainer.global_step))

            if main_process and logs:
                mean_return = float(logs.get("environment/episode_return", 0.0))
                mean_success = float(logs.get("environment/success", 0.0))
                if writer is not None:
                    writer.add_scalar("train/mean_return", mean_return, env_steps)
                    writer.add_scalar("train/mean_success", mean_success, env_steps)
                    for key, value in logs.items():
                        if key.startswith("losses/"):
                            writer.add_scalar(key.replace("losses/", "loss/"), float(value), env_steps)

                append_metrics_csv(
                    artifacts_dir / "logs" / "train_metrics.csv",
                    {
                        "env_steps": env_steps,
                        "mean_return": mean_return,
                        "mean_success": mean_success,
                        **{
                            key.replace("losses/", ""): float(value)
                            for key, value in logs.items()
                            if key.startswith("losses/")
                        },
                    },
                )

                save_checkpoint(
                    artifacts_dir / "checkpoints" / "latest.pt",
                    model=base_model,
                    optimizer=trainer.optimizer,
                    step=env_steps,
                    metadata={
                        "adapter": adapter_info.backend_name,
                        "pufferlib_version": adapter_info.pufferlib_version,
                        "pufferl_available": adapter_info.pufferl_available,
                        "world_size": world_size,
                    },
                )

            timeout_reached = False
            if args.max_runtime_minutes is not None:
                timeout_reached = time.monotonic() - start_time >= args.max_runtime_minutes * 60.0
            if synchronize_stop(timeout_reached, device):
                if main_process:
                    print(f"Stopping early after reaching the runtime limit of {args.max_runtime_minutes} minutes.")
                break

            if env_steps >= next_eval:
                barrier()
                if main_process:
                    summary = evaluate_policy(
                        env_cfg,
                        base_model,
                        device,
                        episodes=int(train_cfg["evaluation"]["episodes"]),
                        seed_start=int(train_cfg["env"]["eval_seed_start"]),
                        difficulty="easy",
                        save_videos=True,
                        video_dir=artifacts_dir / "videos" / f"step_{env_steps}",
                        fixed_video_seeds=list(train_cfg["env"]["eval_fixed_seeds"])[: int(train_cfg["evaluation"]["video_episodes"])],
                    )
                    append_metrics_csv(artifacts_dir / "logs" / "eval_metrics.csv", {"env_steps": env_steps, **summary})
                    if writer is not None:
                        writer.add_scalar("eval/reward_mean", summary["reward_mean"], env_steps)
                        writer.add_scalar("eval/success_rate", summary["success_rate"], env_steps)
                    save_json(artifacts_dir / "eval" / f"summary_{env_steps}.json", summary)
                    eval_steps.append(env_steps)
                    eval_rewards.append(summary["reward_mean"])
                    eval_successes.append(summary["success_rate"])
                    plot_training_curves(artifacts_dir / "plots" / "training_curves.png", eval_steps, eval_rewards, eval_successes)
                    if summary["success_rate"] >= max(eval_successes):
                        save_checkpoint(
                            artifacts_dir / "checkpoints" / "best.pt",
                            model=base_model,
                            optimizer=trainer.optimizer,
                            step=env_steps,
                            metadata=summary,
                        )
                barrier()
                next_eval += eval_interval
    finally:
        close_trainer(trainer, main_process=main_process)
        if writer is not None:
            writer.close()
        if is_distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
