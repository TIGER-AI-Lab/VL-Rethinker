# /*
#  * Original Copyright Haozhe Wang in 2025
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  */
import itertools
import math
import os
import socket
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
import torch.distributed
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset#, SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer, Evaluator
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer, get_vl_processor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_util import init_process_group

from .launcher import BasePPORole
from .utils import get_physical_gpu_id
import json
import shutil


class Evaluator2(Evaluator):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        # self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        # self.critic_train_remote = critic_train_remote
        args = self.strategy.args
        train_data = getattr(args, 'prompt_data',None)
        eval_data = getattr(args, "eval_data",None)
        self.gt_path = [train_data, eval_data]
        print('!!!! gts', self.gt_path)
        self.modelfamily = kwargs.get('modelfamily', 'qwen')
        self.experience_maker = RemoteExperienceMaker(
            None,
            None,
            None,
            None,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            0.0, # self.kl_ctl,
            self.strategy,
            None,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
            gt_path=self.gt_path, 
            modelfamily=self.modelfamily
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        # if backend == "nccl" and self.strategy.args.colocate_all_models:
        #     self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def evaluate(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}
        strategy = self.strategy
        args = self.strategy.args
        eval_dp = getattr(args, "eval_data", None)
        assert eval_dp, f"args.eval_data: {eval_dp} is invalid"
        # if eval_dp: 
        eval_data = blending_datasets(
            eval_dp,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        
        self.eval_data = PromptDataset(
            eval_data, self.tokenizer, strategy, input_template=args.input_template, is_eval=True, processor=self.processor
        )
        print('!!!!! eval data', len(eval_data), eval_dp)
        print(self.eval_data)
        status = super().evaluate(
            args, self.eval_data
        )
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience, global_steps, **kwargs) -> Dict[str, float]:
        return self.training_step_actor(experience, global_steps=global_steps, **kwargs)

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        save_path = None
        print('!!!! [saving] inside actor save_checkpoint')
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor or self.tokenizer,
                save_path,
            )
            max_num = args.max_ckpt_num
            if self.strategy.is_rank_0():
                while True:
                    save_dir = args.ckpt_path 
                    subdirs = sorted(
                        [
                            (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                            for d in os.listdir(save_dir)
                            if d.endswith('hf') and os.path.isdir(os.path.join(save_dir, d)) 
                        ],
                        key=lambda x: x[1],
                    ) # only take folders that ends with hf
                    
                    if len(subdirs) >= max_num: # or total_size > MAX_SIZE:
                        oldest_dir = subdirs[0][0]
                        if os.path.exists(oldest_dir):
                            shutil.rmtree(oldest_dir)
                            print(f"Deleted oldest ckpt {oldest_dir}")
                    else:
                        break
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
                
        return save_path
        


# @ray.remote(num_gpus=1)
# class ActorModelRayActor(BasePPORole):
#     def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
#         args = strategy.args
#         print('====== args ======')
#         print(args)
#         print('==================')
#         if getattr(args, "vllm_num_engines", 0) > 0:
#             # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
#             # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
#             if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
#                 os.environ["NCCL_CUMEM_ENABLE"] = "0"

#         self._setup_distributed(strategy)

#         actor = Actor(
#             pretrain,
#             use_flash_attention_2=strategy.args.flash_attn,
#             bf16=strategy.args.bf16,
#             load_in_4bit=strategy.args.load_in_4bit,
#             lora_rank=strategy.args.lora_rank,
#             lora_alpha=strategy.args.lora_alpha,
#             target_modules=strategy.args.target_modules,
#             lora_dropout=strategy.args.lora_dropout,
#             ds_config=strategy.get_ds_train_config(is_actor=True),
#             packing_samples=strategy.args.packing_samples,
#         )
#         strategy.print(actor)
#         # Support freeze some parameter
#         if hasattr(strategy.args, "freeze_prefix") and strategy.args.freeze_prefix:
#             frozen_count = 0
#             total_params = 0
#             for name, param in actor.model.named_parameters():
#                 total_params += 1
#                 if any(name.startswith(prefix) for prefix in strategy.args.freeze_prefix):
#                     param.requires_grad = False
#                     frozen_count += 1
#             strategy.print(f"Froze {frozen_count}/{total_params} parameters based on prefixes: {strategy.args.freeze_prefix}")

#         # configure tokenizer
#         tokpath = getattr(args, "tokpath", "none")
#         if args.train_vlm:
#             print('!!!! [config] using vlm processor')
#             self.processor = get_vl_processor(
#                 pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
#             )
#             self.tokenizer = self.processor.tokenizer
#         else:
#             self.processor = None
#             self.tokenizer = get_tokenizer(
#                 tokpath if tokpath!='none' else pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
#             )

#         if args.enable_ema:
#             ema_model = Actor(
#                 pretrain,
#                 use_flash_attention_2=strategy.args.flash_attn,
#                 bf16=strategy.args.bf16,
#                 load_in_4bit=strategy.args.load_in_4bit,
#                 ds_config=strategy.get_ds_eval_config(offload=True),
#                 packing_samples=strategy.args.packing_samples,
#             )
#         else:
#             ema_model = None

#         # configure optimizer
#         actor_optim = strategy.create_optimizer(
#             actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
#         )

#         # prepare_datasets
#         self.prepare_datasets()

#         # configure scheduler
#         self.num_update_steps_per_episodes = (
#             len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
#         )
#         print(f"!!!! [config] num_update_steps_per_ep: {args.max_epochs} epoch * {len(self.prompts_dataset)} * {args.n_samples_per_prompt} total-QAs // {args.train_batch_size} (train-bsz QAs) ")
#         max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
#         self._max_steps = max_steps
#         warmup_steps = math.ceil(max_steps * args.lr_warmup_ratio)
#         print(f'!!!! [config] cosine with minlr: maxstep={max_steps},warmupstep={warmup_steps}, lr={args.actor_learning_rate}, minlr=0.1*lr')
#         actor_scheduler = get_scheduler(
#             "cosine_with_min_lr",
#             actor_optim,
#             num_warmup_steps=warmup_steps, 
#             num_training_steps=max_steps,
#             scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.5},
#         )

#         if args.gradient_checkpointing:
#             actor.gradient_checkpointing_enable(
#                 gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
#             )

#         # prepare models/optimizers...
#         self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
#             (actor, actor_optim, actor_scheduler),
#             is_rlhf=True,
#         )

#         if ema_model:
#             ema_model._offload = True
#             self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
#         else:
#             self.ema_model = None

#         # load checkpoint
#         self.consumed_samples = 0
#         ckpt_path = os.path.join(args.ckpt_path, "_actor")
#         if args.load_checkpoint and os.path.exists(ckpt_path):
#             _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
#             self.consumed_samples = states["consumed_samples"]
#             strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

#     def prepare_datasets(self):
#         strategy = self.strategy
#         args = self.strategy.args

#         # prepare datasets
#         prompts_data = blending_datasets(
#             args.prompt_data,
#             args.prompt_data_probs,
#             strategy,
#             args.seed,
#             max_count=args.max_samples,
#             return_eval=False,
#             train_split=args.prompt_split,
#         )
#         prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
#         print('!!!!! prompts', len(prompts_data), args.prompt_data)
#         print(prompts_data)
        
#         self.prompts_dataset = PromptDataset(
#             prompts_data, self.tokenizer, strategy, input_template=args.input_template,
#             processor=self.processor
#         )
#         eval_dp = getattr(args, "eval_data", None)
#         self.eval_data = None
#         if eval_dp: 
#             eval_data = blending_datasets(
#                 eval_dp,
#                 args.prompt_data_probs,
#                 strategy,
#                 args.seed,
#                 max_count=args.max_samples,
#                 return_eval=False,
#                 train_split=args.prompt_split,
#             )
            
#             self.eval_data = PromptDataset(
#                 eval_data, self.tokenizer, strategy, input_template=args.input_template, is_eval=True, processor=self.processor
#             )
#             print('!!!!! eval data', len(eval_data), eval_dp)
#             print(self.eval_data)
        
#         print(f'!!!! RL loader batchsize (queries) = (buffersize) {args.rollout_batch_size} / (worldsize) {strategy.world_size}')
#         self.prompts_dataloader = strategy.setup_dataloader(
#             self.prompts_dataset, args.rollout_batch_size // strategy.world_size, True, shuffle=True,
#         )
#         print('!!!! pretrain', args.pretrain_data)
#         if args.pretrain_data:
#             pretrain_data = blending_datasets(
#                 args.pretrain_data,
#                 args.pretrain_data_probs,
#                 strategy,
#                 args.seed,
#                 return_eval=False,
#                 train_split=args.pretrain_split,
#             )
#             pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
#             pretrain_dataset = SFTDataset(
#                 pretrain_data.select(
#                     range(
#                         min(
#                             len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
#                         )
#                     )
#                 ),
#                 self.tokenizer,
#                 pretrain_max_len,
#                 strategy,
#                 pretrain_mode=True,
#             )
#             self.pretrain_dataloader = itertools.cycle(
#                 iter(
#                     strategy.setup_dataloader(
#                         pretrain_dataset,
#                         args.micro_train_batch_size,
#                         True,
#                         True,
#                         pretrain_dataset.collate_fn,
#                     )
#                 )
#             )
#         else:
#             self.pretrain_dataloader = None

#     def max_steps(self):
#         """Return the maximum number of steps."""
#         return self._max_steps

#     def fit(
#         self,
#         critic_model: ray.actor.ActorHandle,
#         initial_model: ray.actor.ActorHandle,
#         reward_model: List[ray.actor.ActorHandle],
#         remote_rm_url: List[str] = None,
#         reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
#         vllm_engines: List[ray.actor.ActorHandle] = None,
#         critic_train_remote: bool = False,
#     ):
#         """Train actor model with prompt datasets."""
#         strategy = self.strategy
#         args = self.strategy.args

#         # configure Trainer
#         trainer = ActorPPOTrainer(
#             strategy,
#             self.actor,
#             critic_model,
#             reward_model,
#             initial_model,
#             ema_model=self.ema_model,
#             actor_optim=None,
#             critic_optim=None,
#             actor_scheduler=self.actor_scheduler,
#             critic_scheduler=None,
#             remote_rm_url=remote_rm_url,
#             reward_fn=reward_fn,
#             vllm_engines=vllm_engines,
#             max_epochs=args.max_epochs,
#             micro_train_batch_size=args.micro_train_batch_size,
#             micro_rollout_batch_size=args.micro_rollout_batch_size,
#             gradient_checkpointing=args.gradient_checkpointing,
#             critic_train_remote=critic_train_remote,
#             tokenizer=self.tokenizer,
#             processor=self.processor, 
#             prompt_max_len=args.prompt_max_len,
#             value_clip=args.value_clip,
#             eps_clip=args.eps_clip,
#             gamma=args.gamma,
#             lambd=args.lambd,
#             init_kl_coef=args.init_kl_coef,
#             kl_target=args.kl_target,
#             ema_beta=0.992,
#             ptx_coef=args.ptx_coef,
#             max_norm=args.max_norm,
#             # fro GPT generation
#             do_sample=True,
#             max_new_tokens=args.generate_max_len,
#             max_length=args.max_len,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             save_hf_ckpt=args.save_hf_ckpt,
#             disable_ds_ckpt=args.disable_ds_ckpt,
#             gt_path=args.gt_path,
#             modelfamily=args.modelfamily
#         )

#         # broadcast checkpoint
#         ckpt_path = os.path.join(args.ckpt_path, "_actor")
#         if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
#             torch.distributed.barrier()
#             trainer._broadcast_to_vllm()

#         trainer.fit(
#             args,
#             self.prompts_dataloader,
#             self.pretrain_dataloader,
#             self.consumed_samples,
#             self.num_update_steps_per_episodes,
#             eval_data=self.eval_data
#         )

#     def save_model(self):
#         pass 
#         # args = self.strategy.args

#         # # save model checkpoint after fitting on only rank0
#         # self.strategy.save_model(
#         #     self.ema_model if args.enable_ema else self.actor,
#         #     self.tokenizer,
#         #     args.save_path,
#         # )
