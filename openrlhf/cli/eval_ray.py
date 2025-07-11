# /*
#  * Modified by Haozhe Wang in 2025
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  */

import argparse
from datetime import datetime
from typing import List
import os
import socket
import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
    Evaluator2,
)
from openrlhf.utils import get_strategy, get_vl_processor
from openrlhf.trainer.ray.utils import ray_noset_visible_devices


# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)

def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    assert (
        args.rollout_batch_size % actor_world_size == 0
    ), f"rollout_bach_size must be divisible by actor_world_size, got {args.rollout_batch_size} and {actor_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"

    if args.vllm_num_engines > 0:
        assert (
            actor_world_size % args.vllm_num_engines == 0 or args.vllm_num_engines % actor_world_size == 0
        ), f"actor_world_size must be divisible by vllm_num_engines, got {actor_world_size} and {args.vllm_num_engines}"

    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"


def train(args):
    # _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)

    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    # if args.colocate_actor_ref or args.colocate_all_models:
    #     # assert (
    #     #     args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
    #     # ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

    #     bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
    #     pg = placement_group(bundles, strategy="PACK")
    #     ray.get(pg.ready())
    #     print('!!!! [config] using placement groups')

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        # if args.colocate_all_models and args.vllm_gpu_memory_utilization >= 0.9:
        #     args.vllm_gpu_memory_utilization = 0.4
        #     print(
        #         f"Set args.vllm_gpu_memory_utilization to {args.vllm_gpu_memory_utilization} for colocate_all_models!"
        #     )

        #     assert (
        #         args.actor_num_nodes * args.actor_num_gpus_per_node
        #         == args.vllm_num_engines * args.vllm_tensor_parallel_size
        #     ), (
        #         f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
        #         f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
        #         f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
        #     )

        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            args.actor_num_nodes * args.actor_num_gpus_per_node,
            pg if args.colocate_all_models else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
        )

    # actor_model = PPORayActorGroup(
    #     args.actor_num_nodes,
    #     args.actor_num_gpus_per_node,
    #     ActorModelRayActor,
    #     pg=pg,
    #     num_gpus_per_actor=0.2 if pg else 1,
    # )
    actor_model = None 
    ref_model = None
    # if args.init_kl_coef == 0:
    #     ref_model = None
    # else:
    #     ref_model = PPORayActorGroup(
    #         args.ref_num_nodes,
    #         args.ref_num_gpus_per_node,
    #         ReferenceModelRayActor,
    #         pg=pg,
    #         num_gpus_per_actor=0.2 if pg else 1,
    # )

    # if not args.colocate_all_models:
    pg = None

    # if colocated, create placement group for critic and reward model explicitly.
    # if args.critic_pretrain and args.colocate_critic_reward:
    #     assert (
    #         args.critic_num_nodes == args.reward_num_nodes
    #         and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
    #     ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

    #     bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
    #     pg = placement_group(bundles, strategy="PACK")
    #     ray.get(pg.ready())

    # if args.critic_pretrain:
    #     critic_model = PPORayActorGroup(
    #         args.critic_num_nodes,
    #         args.critic_num_gpus_per_node,
    #         CriticModelRayActor,
    #         pg=pg,
    #         num_gpus_per_actor=0.2 if pg else 1,
    #     )
    # else:
    #     critic_model = None
    critic_model = None
    reward_models = None
    # multiple reward models
    # if args.reward_pretrain:
    #     reward_pretrains = args.reward_pretrain.split(",")
    #     reward_models = []
    #     for _ in reward_pretrains:
    #         reward_models.append(
    #             PPORayActorGroup(
    #                 args.reward_num_nodes,
    #                 args.reward_num_gpus_per_node,
    #                 RewardModelRayActor,
    #                 pg=pg,
    #                 num_gpus_per_actor=0.2 if pg else 1,
    #             )
    #         )
    # else:
    #     reward_models = None

    # init reference/reward/actor model
    # refs = []
    # if ref_model is not None:
    #     refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    # refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    # if args.reward_pretrain:
    #     for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
    #         refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    # ray.get(refs)

    # if args.critic_pretrain:
    #     # critic scheduler initialization depends on max_step, so we have to init critic after actor
    #     # TODO: use first reward model as critic model
    #     max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    #     refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
    #     ray.get(refs)

    # train actor and critic mdoel
    master_addr = None
    master_port = None 
    world_size = 1
    rank = 0
    _master_addr = master_addr if master_addr else _get_current_node_ip()
    _master_port = master_port if master_port else _get_free_port()
    os.environ["MASTER_ADDR"] = _master_addr
    os.environ["MASTER_PORT"] = str(_master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    
    os.environ["LOCAL_RANK"] = "0" # str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
    strategy.setup_distributed()
    pretrain = args.pretrain
    processor = get_vl_processor(
                pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
    )
    tokenizer = processor.tokenizer
    evaluator = Evaluator2(
            strategy,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=None, # critic_train_remote,
            tokenizer=tokenizer,
            processor=processor, 
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
            gt_path=args.gt_path,
            modelfamily=args.modelfamily
        )
    
    evaluator.evaluate(0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--param_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")
    parser.add_argument("--train_vlm", action="store_true", default=False)
    parser.add_argument("--freeze_prefix", type=str, nargs="+", default=None,
        help="List of parameter name prefixes to freeze during training"
    )
    parser.add_argument("--drop_maxlen", action="store_true", default=False)

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo","grpo","gloo","rloo_sft","group","group_sft"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument("--eval_data", type=str, default=None, help="HF dataset name or path")
    
    parser.add_argument("--training_mode", type=str, default="train", help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)
    
    # additional 
    parser.add_argument("--gt_path", type=str, default=None)
    parser.add_argument("--modelfamily", type=str, default='qwen')
    parser.add_argument("--system_prompt", type=str, default='None')
    parser.add_argument("--sample_log_file", type=str, default='')
    parser.add_argument("--format", type=str, default='')
    parser.add_argument("--val_temperature", type=float, default=0.6)
    parser.add_argument("--curriculum_filter", type=str, default=None)
    parser.add_argument("--curriculum_ncollection", type=int, default=2)
    parser.add_argument("--tokpath", type=str, default="none")
    parser.add_argument("--rule_reward", type=str, default="default")
    parser.add_argument("--prefix_generation", type=str, default="default")
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--data_version", type=str, default="default")
    parser.add_argument("--loss_version", type=str, default="default")
    parser.add_argument("--buffer_norm", type=int, default=0)
    parser.add_argument("--think_only", type=int, default=0)
    parser.add_argument("--kl_penalty_coef", type=float, default=0.0)
    parser.add_argument("--entropy_loss_coef", type=float, default=0.0)
    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if args.reward_pretrain:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo","rloo_sft"]:
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.vllm_num_engines >= 1 and args.enable_prefix_caching:
        args.enable_prefix_caching = False
        print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache.")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.train_vlm: 
        if args.packing_samples:
            print("[Warning] --train_vlm is not supported with --packing_samples. We will set args.packing_samples to False")
            args.packing_samples = False
        if args.pretrain_data:
            print("[Warning] --train_vlm is not supported with --pretrain_data. We will set args.pretrain_data to None")
            args.pretrain_data = None

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
