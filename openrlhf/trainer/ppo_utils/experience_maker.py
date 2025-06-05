# /*
#  * Modified by Haozhe Wang in 2025
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  */

import os
import time
from abc import ABC
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

import ray
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import interleave_datasets, load_dataset
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
# from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from collections import defaultdict

import datasets
import json
# pip install math-verify
from math_verify import parse, verify
import pickle as pkl
import re 
from PIL import Image

logger = init_logger(__name__)


def extract_qwen_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("<|im_start|>assistant\n")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("<|im_start|>user\n")[1].split('<|im_end|>')[0].split('<|vision_end|>')[-1]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response


def extract_dsmath_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("Assistant:")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("User:")[1].strip()
    
    # Return the user query and the assistant's response
    return user_query, assistant_response


def extract_dpsk_query_and_response(input_text):
    # Split the input text by the assistant's start token
    # print(input_text)
    parts = input_text.split("<｜Assistant｜>")
    
    # The first part contains the system and user messages
    if len(parts)==0:
        print('!!!! warning extraction', input_text)
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("<｜User｜>")[1]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response

def extract_llama_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("assistant<|end_header_id|>\n\n")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("user<|end_header_id|>\n\n")[1].split('<|eot_id|><|start_header_id|>')[0]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response

def extract_autocode_query_and_response(input_text):
    # print('!!!! example input', input_text)
    # Split the input text by the assistant's start token
    parts = input_text.split("Response:")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("### Instruction:\n")[1].split('\n\n### ')[0]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response
    
def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    visual_inputs: Optional[dict] = field(default_factory=dict)
    validity: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) if isinstance(value, torch.Tensor) else value for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: to(value, device) for key, value in self.visual_inputs.items()}
        if self.validity is not None: 
            self.validity = to(self.validity, device)
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) if isinstance(value, torch.Tensor) else value for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: pin_memory(value) for key, value in self.visual_inputs.items()}
        if self.validity is not None: 
            self.validity = pin_memory(self.validity)
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    visual_inputs: the visual input for vlm training
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    visual_inputs: Optional[Dict]
    na_each: list[int]
    round0_correctness: list 
    round1_correctness: list 
    round0_nwait: list[int]
    round1_nwait: list[int]
    questions: list[str]
    solutions: list[str]
    qids: list[str]
    round0_ALLTrue: list[float]
    round0_Easy: list[float]
    round0_Medium: list[float]
    round0_Hard: list[float]
    round0_ALLFalse: list[float]
    # round0_saturation: list[float]

def get_raw(modelfamily, text):
    if modelfamily=='dpsk':
        user = text.split("<｜Assistant｜>")[0].split("<｜User｜>")[1]
        return user
    
class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        data_processor,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
        modelfamily='qwen',
        gt_path=None
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.gt_path = gt_path
        self.modelfamily = modelfamily
        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        pass 

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            if self.data_processor is not None:
                inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs = {}
                for k,v in inputs.items():
                    if k not in ["input_ids", "attention_mask"]:
                        visual_inputs[k] = v
            else:
                inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
                visual_inputs = None

            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                visual_inputs=visual_inputs,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def get_logprobs_and_logs(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        pass

    @torch.no_grad()
    def handle_advantages(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        print(f"===> [verbose] handling advantages in NaiveEMaker handle_advantages()")
        do_longer = getattr(args, "format", "none") == 'longer'
        tmp = [experience.info["reward"] for experience in experiences]
        ns =  [experience.info["response_length"]  for experience in experiences]
        match = [experience.info["match"]  for experience in experiences]
        ns = np.array(ns).reshape((-1, args.n_samples_per_prompt)) 
        match = np.array(match).reshape((-1, args.n_samples_per_prompt)) 
        ns_diff = []
        for idx, match_i in enumerate(match): 
            # when there is no correct 
            if np.sum(match_i)==0: 
                ns_diff.append(np.zeros_like(ns[idx]))
            else: 
                mean_switch = np.sum(ns[idx] * match_i)/np.sum(match_i) # average length of correct response 
                len_adv = (ns[idx]-mean_switch)*(match_i>0.5) # positive values of longer 
                max_adv = abs(max(len_adv)) # right delta
                min_adv = abs(min(len_adv)) # left delta 
                # if min_adv<0: min_adv = -min_adv
                len_adv[len_adv>0] /= max_adv # normalized to [-1.0, 1.0]
                len_adv[len_adv<0] /= min_adv
                ns_diff.append(len_adv)
                # tmplist = []
        ns_diff = np.stack(ns_diff)
        bonus = np.clip(ns_diff * 1.0, -0.499, 0.499)
        num = len(experiences)
        bonus_flat = bonus.reshape((num, -1))
        for idx, exp in enumerate(experiences):
            exp.info["wait_bonus"] = bonus_flat[idx].tolist()
        print(f'!!!! [rbuffer] The estimator {args.advantage_estimator} is processing {len(experiences)} queries in a batch, each {len(tmp[0])} responses, longer={do_longer}')
        # reward shaping for RLOO
        if args.advantage_estimator in ["rloo","gloo","rloo_sft"]: # this operates in batch level
            rewards = torch.cat(tmp) 
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")  # (bsz,nsample) into groups
            
            if do_longer:
                bonus_tensor = torch.from_numpy(bonus).to(rewards.device).to(rewards.dtype)
                rewards += bonus_tensor
                # print('!!!! shaped reward', rewards.detach().cpu().numpy())
            else:
                print('!!!! [rbuffer] reward not using wait')
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1) # mean of others 
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator in ["group", "group_sft"]: # this operates in batch level
            rewards = torch.cat(tmp) 
            rewards = rewards.reshape(-1, args.n_samples_per_prompt) # .to(device="cuda")  # (bsz,nsample) into groups
            raw_r = rewards.detach().numpy() # bsz,nsamples 
            mean_acc = np.tile(raw_r.mean(-1, keepdims=True), (1,args.n_samples_per_prompt))
            solve_all = mean_acc>0.95
            solve_none = mean_acc<0.05
            easy = mean_acc>0.7
            hard = mean_acc<0.35
            medium = np.logical_not(np.logical_or(easy, hard))
            
            difficulty, solve_all, solve_none, easy, hard, medium = [x.reshape((len(experiences), -1)).astype(float) for x in [mean_acc, solve_all, solve_none, easy, hard, medium]]
            all_waits = []
            all_waits0 = []
            is_native = []
            t1_diff = []
            for iidx, exp in enumerate(experiences):
                exp.info['difficulty'] = difficulty[iidx].tolist()
                exp.info['solve_all'] = solve_all[iidx].tolist()
                exp.info['solve_none'] = solve_none[iidx].tolist()
                exp.info['easy'] = easy[iidx].tolist()
                exp.info['hard'] = hard[iidx].tolist()
                exp.info['medium'] = medium[iidx].tolist()
                all_waits.extend(exp.info['round1_nwait'])
                all_waits0.extend(exp.info['round0_nwait'])
                t1_cor = exp.info['round1_correctness']
                t0_cor = exp.info['round0_correctness']
                is_native.extend([float(x is None) for x in t1_cor])
                t1_diff.extend([-5.0 if x<0 else x-y for x,y in zip(t1_cor, t0_cor) ]) # x=-1 if no rethinking
                # print('!!!! [debug] solve status', exp.info['solve_all'], raw_r.mean(-1))
            reshaped_nwait_round1 = np.array(all_waits).reshape((len(rewards), -1))
            reshaped_nwait_round0 = np.array(all_waits0).reshape((len(rewards), -1))
            reshaped_is_native = np.array(is_native).reshape((len(rewards), -1))
            reshaped_t1_diff = np.array(t1_diff).reshape((len(rewards), -1))
            # all_waits = (np.logical_and(reshaped_nwait_round1>0, reshaped_nwait_round1<=2)).astype(float)
            # too_many_waits = (reshaped_nwait_round1>2).astype(float)
            # all_waits = torch.from_numpy(all_waits).to(rewards.device)
            baseline = rewards.sum(-1, keepdim=True) / (args.n_samples_per_prompt) # mean of others 
            rewards = rewards - baseline
            
            # for iidx in range(len(rewards)):
            #     if reshaped_nwait_round1[iidx].sum()>0: 
            #         # if advantage>0.125, meaning this is informative positive example 
            #         # - if it has native wait or keep into a correct response, we praise it
            #         isnative = reshaped_is_native[iidx]>0.5
            #         ischeck = reshaped_t1_diff[iidx]==0.0
            #         notmanywaits = reshaped_nwait_round1[iidx]<4.0
            #         old = rewards[iidx].cpu().numpy()
            #         oldstr = str(old)
            #         # praise = torch.ones_like(rewards[iidx])
            #         # praiseflag = np.logical_and(notmanywaits,np.logical_or(isnative, ischeck))
            #         praiseflag = np.logical_and(notmanywaits,np.logical_or(isnative, ischeck))
            #         flag = False
            #         # print(f"[debug] native={isnative}, {reshaped_is_native[iidx]}, check={ischeck}, {reshaped_t1_diff[iidx]}, wait={notmanywaits}, {reshaped_nwait_round1[iidx]}")
            #         for ii,(rvalue,pflag,wflag,macc) in enumerate(zip(rewards[iidx], praiseflag, notmanywaits, baseline[iidx])):
                        ################
                        # if pflag and rvalue>0.: # correct and native wait
                        #     rewards[iidx][ii] = rvalue * 1.5
                        #     flag = True
                        # elif rvalue<0.0 and wflag:
                        #     rewards[iidx][ii] = rvalue * 0.5
                        #     flag = True 
                        #################
                        # if macc>0.98:
                        #     rewards[iidx][ii] = 1.0/8.0
                            
                    # scale_factor = (rewards[iidx]> 0.1 ).to(float) * praise 
                    # mask = 1.0+scale_factor # torch.tensor 
                    # - if a incorrect has a bit of rethinking try, we praise it 
                    # selector = torch.BoolTensor(notmanywaits).to(rewards.device)
                    # mask[selector] = 1.0 - 0.5*(rewards[iidx] < 0.0).to(float)
                    # new = rewards[iidx] * mask
                    # - if a incorrect has a bit of rethinking try, we praise it 
                    # scale_factor = (rewards[iidx]> 0.1 ).to(float) * all_waits[iidx] # (nsamples,) is positive and has wait? 
                    # if too_many_waits[iidx].sum()>0:
                    #     for kk, entry in enumerate(too_many_waits[iidx]):
                    #         if entry>0.5: 
                    #             scale_factor[kk] = -1.
                    #             print('!!!! [debug] too many waits supressed.')
                    # old = rewards[iidx].cpu().numpy()
                    # new = rewards[iidx] * (1.0+scale_factor) # positive examples with wait will get x2 advantages
                    # if flag: print(f'!!!! [debug] {oldstr} ->{rewards[iidx].cpu().numpy()}')
                    # rewards[iidx] = new
            
            if do_longer:
                print('!!!! length bonus', bonus)
                bonus_tensor = torch.from_numpy(bonus).to(rewards.device).to(rewards.dtype).reshape(rewards.shape)
                rewards  = (bonus_tensor+1)*rewards
            # else:
                # print('!!!! not using wait')
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences)) # num_exp, 
            return experiences, rewards
        # default rewards
        return experiences, tmp

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

def regularize_text(x):
    trigger = "Please reason step by step, and put your final answer within \\boxed{}."
    x = x.split(trigger)[0]
    return x.strip().replace(' ','')

def do_verify(nsol, b):
    res = 0.0 
    try:
        a = parse(nsol)
        if len(b)>1 and (b[1] in 'ABCDEFGHIJK'):
            res = float(nsol[len("\\boxed{"):].startswith(b[1]))
        else:
            # print(f"debug parsed: {a} from {nsol} and {b}")
            if len(a)==0: res = -1.0 
            else: res = float(verify(a, b))
    except: 
        print(f"!!!! [debug] {nsol} parsing exception")
        res = -1.0 
    return res 

ans_indicator = "answer is"
endstr = "Now everything looks fine. Solution finished."
def normalize_answer(answer):
    if answer is None: return answer
    if 'dfrac' in answer: answer = answer.replace("dfrac", "frac")
    # if '%' in answer: answer = answer.replace(r'\%',"").replace('%',"")
    if 'text' in answer: answer = answer.replace("\\text","")
    if "\\varnothing" in answer: answer = answer.replace("\\varnothing","\\emptyset")
    if "minutes" in answer: answer = answer.replace("minutes","")
    if "cm" in answer: answer = answer.replace("cm","")
    # if "^\\circ" in answer: answer = answer.replace("^\\circ","")
    # if "a.m." in answer: answer = answer.replace("a.m.","")
    return answer 

def handle_boxed(sol, gt, eostoken, format_type, requires_box=False):
    # print(sol)
    # print('!!!! debug', gt)
    norepeat = None
    usefmt = None
    res = 0.0
    # endstr and eos token
    index = sol.find(endstr)
    num_end = len(eostoken)
    
    if index>-1: 
        remains = sol[index+len(endstr)+num_end:]
        if len(remains)>0: 
            norepeat = False 
        else: norepeat = True 
    if not (norepeat is False):
        if format_type in ["confidence"]:
            if not ("<confidence>" in sol and "</confidence>" in sol):
                usefmt = False
            else: 
                count = sol.count("<confidence>")
                if count>5: usefmt = False 
                else: usefmt = True 
        elif format_type in ["wait"]:
            tmps = sol.lower()
            usefmt = False 
            if "wait" in tmps or "alternatively" in tmps:
                usefmt = True 
            
        # elif format_type in ["nocode"]:
        #     if "```python" in sol:
        #         usefmt = False 
        #     else: usefmt = True 
    
    if (norepeat is False): 
        pass # no need for correctness 
    else: 
        flag = True 
        gt = normalize_answer(gt)
        try:
            if "\\boxed" in gt: 
                b = parse(gt)
                # print('!!!! debug gt parse', gt, b)
            else:
                b = parse(f"\\boxed{{{gt}}}")
        except Exception as e:
            print(f"!!!! [debug] {gt} parsing exception")
            res = -1.0
            flag = False 
        if flag:
            if len(b)==0: res = -1.0 
            else: 
                if requires_box:
                    boxed_index = sol.rindex("boxed")
                    if boxed_index==-1: res = 0.0 
                    else:
                        nsol = '\\'+sol[boxed_index:]
                        res = do_verify(normalize_answer(nsol), b)
                else: 
                    flag = False 
                    
                    for indicator in ["\\boxed", "<answer>", "Answer:"]:
                        if indicator in sol:
                            if indicator == "<answer>":
                                found = re.search("<answer>(.*?)</answer>", sol)
                                if found:
                                    nsol = f"\\boxed{{{found.group(1)}}}"
                                else: continue 
                            elif indicator == "Answer:":
                                tmp = sol.split(indicator)
                                if len(tmp)>0: tmp = tmp[-1].strip() # .split(eostoken)[0].strip()
                                else: continue 
                                nsol = f"\\boxed{{{tmp}}}"
                            else: 
                                boxed_index = sol.rindex(indicator)
                                pred = sol[boxed_index:].strip()
                                nsol = pred
                            res = do_verify(normalize_answer(nsol), b)
                            if res > 0.99: 
                                flag = True 
                        if flag: 
                            break
                    # print("extracted sol", nsol)
                    if not flag:
                        nsol = sol 
                        res = do_verify(normalize_answer(nsol), b)
        
                    
    return norepeat, usefmt, res 

def rule_reward(sol, gt, eostoken, format_type, requires_box, *args):
    # valid = eos & boxed 
    error_info = None 
    valid = True 
    if eostoken not in sol and "<|endoftext|>" not in sol: 
        valid = False
        error_info = "No eos."
    elif requires_box and "boxed" not in sol:
        valid = False 
        error_info = "No valid boxed."
    elif sol.lower().count("wait")>5:
        valid = False
        error_info = "too many waits"
    ############ this is only for debugging 
    # if format_type=='wait':
    # tmps = sol.lower() 
    # if "wait" not in tmps and "alternatively" not in tmps:
    #     valid = False 
    
    # formats and correctness 
    norepeat = None
    usefmt = None
    res = 0.0
    # directly making no-boxed and no-eos as invalid seems unnecessary and harmful:
    # the model needs to understand these are unacceptable
    # if not valid: 
    #     pass 
    # else: 
    norepeat, usefmt, res = handle_boxed(sol, gt, eostoken, format_type, requires_box=requires_box)
        
    return valid, norepeat, usefmt, error_info, res 


def batch_rule_reward(sols, gts, eostoken, format_type, *args):
    rets = []
    for sol, gt in zip(sols,gts):
        rets.append(rule_reward(sol, gt, eostoken, format_type, *args))
    return rets


def find_last_code_block(text):
    # Define the regex pattern to match code blocks enclosed with ```python and ```
    pattern = r'```python(.*?)```'
    
    # Reverse the text and the pattern
    reversed_text = text[::-1]
    
    reversed_pattern = r'```(.*?)nohtyp```'
    
    # Search for the reversed pattern in the reversed text
    match = re.search(reversed_pattern, reversed_text, re.DOTALL)
    
    if match:
        # Extract the matched group, reverse it back to get the original code block
        reversed_code_block = match.group(1).strip()
        code_block = reversed_code_block[::-1]
        return code_block
    else:
        return None


def rule_reward_with_code(sol, gt, eostoken, format_type, executor):
    error_info = None 
    # valid = eos & boxed 
    valid = True 
    # formats and correctness 
    norepeat = None
    usefmt = None
    res = 0.0
    if eostoken not in sol: 
        valid = False
        return valid, norepeat, usefmt, error_info, res 
    if "```python" in sol:
        code = find_last_code_block(sol)
        if code is None: # no code found 
            valid = False 
            error_info = "No valid code block."
            return valid, norepeat, usefmt, error_info, res 
        pred, error_info = executor.apply(code)
        if error_info=='Done':
            try:
                b = parse(f"\\boxed{{{gt}}}")
                
                nsol = '\\boxed{'+pred+'}'
                a = parse(nsol)
                
                if len(a)==0: res = -1.0 
                else: res = float(verify(a, b))
            except:
                res = -1.0 
            error_info += f": {pred}"
        else: res = 0.0 
        # print(res, pred, error_info)
    else:
        if "boxed" not in sol:
             valid = False 
             return valid, norepeat, usefmt, "No valid boxed.", res 
        
        norepeat, usefmt, res = handle_boxed(sol, gt, eostoken, format_type)
    return valid, norepeat, usefmt, error_info, res 

def batch_rule_reward_with_code(sols, gts, eostoken, format_type, executor, requires_box=False):
    rets = []
    codes, code_i = [],[]
    # print('!!!! inside reward requires box', requires_box)
    for ii,(sol,gt) in enumerate(zip(sols, gts)):
        error_info = None 
        # valid = eos & boxed 
        valid = True 
        # formats and correctness 
        norepeat = None
        usefmt = None
        res = 0.0
        usecode = None 
        if eostoken not in sol: 
            valid = False
            ret = valid, norepeat, usefmt, error_info, usecode, res 
            rets.append(ret)
            # print('!!!! not valid: no eos', sol)
            continue
        if "```python" in sol:
            code = find_last_code_block(sol)
            usecode = True 
            if code is None: # no code found 
                valid = False 
                # print('!!!! not valid: no code', sol)
                error_info = "No valid code block."
                ret = valid, norepeat, usefmt, error_info, usecode, res 
                rets.append(ret)
                continue 
            codes.append(code)
            code_i.append(ii)
            ret = valid, norepeat, usefmt, error_info, usecode, res 
            rets.append(ret)
            continue 
            
        else:
            usecode = False 
            if requires_box and ("boxed" not in sol):
                valid = False 
                ret = valid, norepeat, usefmt, "No valid boxed.", usecode, res 
                rets.append(ret)
                continue 
            
            norepeat, usefmt, res = handle_boxed(sol, gt, eostoken, format_type, requires_box=requires_box)
            ret = valid, norepeat, usefmt, error_info, usecode, res 
            rets.append(ret)
            continue 
        
        if format_type in ['nocode']:
            if '```python' in sol: usefmt = False 
            else: usefmt = True
        
    #####
    if len(codes)>0:
        tmp = [executor.apply(c) for c in codes]
        preds, error_infos = list(zip(*tmp))
        for ii,code,pred,error_info in zip(code_i,codes,preds,error_infos):
            if error_info=='Done':
                flag = True 
                try:
                    gt = gts[ii]
                    b = parse(f"\\boxed{{{gt}}}")
                except:
                    res = -1.0 
                    flag = False 
                if flag: 
                    nsol = pred
                    res = do_verify(nsol, b)
                error_info += f": {pred}"
            else: res = 0.0 
            valid, norepeat, usefmt, _, usecode, _ = rets[ii]
            rets[ii] = valid, norepeat, usefmt, error_info, usecode, res 
    
    return rets
        
def prepare_target(prompt, eos_token):
    if "</think>" in prompt: 
        tmp = prompt.split("</think>")[0]+"</think>" 
        # print('!!!! prepare', [tmp])
        return tmp + eos_token
    else: return prompt 
    
def handle_placeholders(texts):
    newlist = []
    placeholder = "<image>"
    # placeholder2 = "<image1>"
    replacewith = "<|vision_start|><|image_pad|><|vision_end|>"
    for m in texts:
        new = m 
        for k in ["<|vision_start|>","<|image_pad|>","<|vision_end|>"]:
            new = new.replace(k,"")
        # now new has no replacewith 
        if new.count(placeholder)>0:
            new = new.replace(placeholder, replacewith)
        else: 
            new = replacewith + new
        newlist.append(new)
    return newlist
        
class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.rule_reward_func = batch_rule_reward
        self.q2gt = dict() 
        self.q2r = defaultdict(list)
        for dp in self.gt_path:
            # dp = gt_path
            if dp is None: continue 
            print('!!!! adding gts for', dp)
            ext = dp.split('.')[-1]
            if ext in ["json", "jsonl", "csv"]:
                ext = ext.lower().strip(".")
                if ext == "jsonl":
                    ext = "json"
                data = datasets.load_dataset(ext, data_files=dp)
                self.qkey = 'question'
                self.gt_key = 'gt_answer'
                
            else:
                if dp.endswith('parquet'): data = load_dataset('parquet', data_files=dp)
                else: data = load_dataset(dp)
                # blending_datasets(dp, "1.0", self.strategy, )
                # data = datasets.load_dataset('parquet', data_dir=dp)
                self.qkey = 'question'
                self.gt_key = 'answer'
            self.qidkey = 'qid'
                
            full_list = []
            for k,v in data.items(): 
                full_list.extend(v.to_list())
            data = full_list
            
            # q2gt
            # q2gt = dict() 
            # do we need to regularize the question?
            for item in data: 
                self.q2gt[item[self.qidkey]] = item[self.gt_key]
                if 'responses' in item: 
                    self.q2r[item[self.qidkey]].extend(item['responses'])
        dataver = getattr(self.strategy.args, "data_version", "red")
        if 'use_response' in dataver:
            assert len(self.q2r)>0, "no q2responses for red mode."
        
        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)
        self.parse_code = False 
        self.executor = None

    def separate_qa(self, queries):
        if self.modelfamily=='qwen':
            return list(zip(*[extract_qwen_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='llamasft':
            return list(zip(*[extract_llama_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='autocode':
            return list(zip(*[extract_autocode_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='dpsk':
            return list(zip(*[extract_dpsk_query_and_response(qq) for qq in queries]))
        elif self.modelfamily=='dsmath':
            return list(zip(*[extract_dsmath_query_and_response(qq) for qq in queries]))
        else:
            raise Exception('Not implemented')
        
    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], is_eval=False, **generate_kwargs) -> List[Experience]:
        print("===> [verbose] remoteEMaker make_experience_list()")
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        self.eval_step = generate_kwargs.get("eval_step", 0)
        args = self.strategy.args
        generate_kwargs['is_eval'] = is_eval
        data_version = getattr(args, "data_version", None)
        if ('use_response' in data_version) and not is_eval:
            samples_list = self.generate_samples(all_prompts, use_response=True, **generate_kwargs)
        else:
            samples_list = self.generate_samples(all_prompts, **generate_kwargs)
        
        print(f"===> [verbose] REMaker get_experience(): single experience is arranged as {args.micro_rollout_batch_size} qas, and nsample={args.n_samples_per_prompt}")
        experiences = []
        for batched_sample in samples_list:
            tmp = self.get_logprobs_and_logs(batched_sample, is_eval=is_eval, validity=None)
            experiences.append(tmp.to_device("cpu"))
        
        experiences, rewards = self.handle_advantages(experiences)
        
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            if experience.action_log_probs is None: continue 
            
            # experience = experience.to_device("cuda")
            # reward = reward.to(device="cuda") # tensor of shape (queries,)
            num_actions = experience.info["num_actions"] # list of shape (queries,)
            ###########
            # kl = [[x, x, x],
            #       [x, x, x]]
            # reward = [1.0, 0.0]
            # reward = [[x,x,x+1.0],
            #           [x,x,x+0.0]]
            reward = compute_reward(
                reward, # tensor of shape (queries,)
                self.kl_ctl.value,
                experience.kl, # list of tensor, each shape = (ntokens,)
                action_mask=experience.action_mask, # None
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            ) # list of tensor, each shape (ntokens,)
            
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo","rloo_sft","group","group_sft"]: # indeed not doing anything
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
                experience.info["return"] = [x.mean() for x in experience.advantages]
                
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            experience.kl = None
            del experience.info["num_actions"]
            # experience.to_device("cpu")
    
        
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        print("!!!! [rbuffer] rearranged as (bsz, nsample) to compute rewards")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[Dict], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        print("===> [verbose] remoteEMaker generate_samples() using generate_vllm()")
        samples = self._generate_vllm(all_prompts, **generate_kwargs)
        print(f"===> [verbose] remoteEMaker generate_samples() done with {len(samples)} samples each with args.micro_rollout_batch_size qas")
        # vLLM offload when colocate_all_models
        if self.strategy.args.vllm_enable_sleep:
            if torch.distributed.get_rank() == 0:
                refs = []
                for engine in self.vllm_engines:
                    refs.append(engine.sleep.remote())
                ray.get(refs)
        return samples

    def convenient_get_batch_rewards_from_queries(self, queries, potential_qids, no_question=False):
        if no_question: solutions = queries
        else:
            questions, solutions = self.separate_qa(queries)
        gts = [self.q2gt.get(q, None) for q in potential_qids]
        format_type = getattr(self.strategy.args, "format", None)
        sysprompt = getattr(self.strategy.args, "system_prompt", None)
        requires_box = False if self.parse_code or sysprompt=='dpsk' else True # sysprompt !='autocode'
        rets = self.rule_reward_func(solutions, gts, self.tokenizer.eos_token, format_type, self.executor, requires_box)
        return rets 
        
    @torch.no_grad()
    def get_logprobs_and_logs(self, batched_sample: Samples, is_eval=False, validity=None, ) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        dataver = getattr(args, "data_version", "red")
        use_response = 'use_response' in dataver
        if self.actor: self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = batched_sample.sequences
        attention_mask = batched_sample.attention_mask
        action_mask = batched_sample.action_mask
        num_actions = batched_sample.num_actions
        na_each = batched_sample.na_each 
        packed_seq_lens = batched_sample.packed_seq_lens
        visual_inputs = batched_sample.visual_inputs
        prompts = batched_sample.prompts
        round0_correctness = batched_sample.round0_correctness
        round1_correctness = batched_sample.round1_correctness
        round0_nwait = batched_sample.round0_nwait 
        round1_nwait = batched_sample.round1_nwait 
        questions = batched_sample.questions 
        solutions = batched_sample.solutions
        potential_qids = batched_sample.qids
        
        num_seq = len(sequences) # default to cpu device
        # potential_qids = []
        # for p in prompts: 
        #     info = json.loads(p)
        #     if 'qid' in info[-1]: # last entry has a key qid
        #         qid = info[-1]['qid']
        #         potential_qids.append(qid) # 768
        
        start = time.time()
        device = 'cuda'
        sequences_cpu, attention_mask_cpu = (
            sequences.to(device),
            attention_mask.to(device),
        )
        visual_inputs_cpu = None
        if visual_inputs is not None:
            visual_inputs_cpu = {k: v.to(device) for k, v in visual_inputs.items()}        
        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens,visual_inputs=visual_inputs_cpu
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs = []
        # if self.reward_model: 
        #     for rm in self.reward_model:
        #         r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu))
        # elif self.remote_rm_url:
        #     if self.custom_reward_func:
        #         r = self.custom_reward_func.remote(queries, batched_sample.prompts)
        #         r_refs.append(r)
        #     else:
        #         for rm in self.remote_rm_url:
        #             r = remote_rm_fn_ray.remote(rm, queries=queries, prompts=batched_sample.prompts)
        #             r_refs.append(r)
        # else:
            # pass 
            # remote RM
            # if not self.packing_samples:
            #     queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            # else:
            #     sequences_list = []
            #     offset = 0
            #     tokens_list = sequences_cpu.tolist()[0]
            #     for length in packed_seq_lens:
            #         sequences_list.append(tokens_list[offset : offset + length])
            #         offset += length
            #     queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
            # pad_token = self.tokenizer.pad_token
            # eos_token = self.tokenizer.eos_token
            # if pad_token==eos_token:
            #     queries = [x.replace(self.tokenizer.pad_token, "")+eos_token for x in queries]
            # else:
            #     queries = [x.replace(self.tokenizer.pad_token, "") for x in queries]
            
        if args.colocate_all_models and self.reward_model:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        acc_rewards = []
        norepeat_rewards = []
        usefmt_rewards = []
        raw_rewards = []
        initial_validity = validity
        validity = None
        error_infos = []
        use_codes = []
        # ns_in_correct = []
        exceptions = []
        eostoken = self.tokenizer.eos_token
        data_version = getattr(args, "data_version", None)
        force_wait = "force_append_wait" in data_version
        if not (self.reward_model or self.remote_rm_url): 
            
            # print('========= using local rule reward =======')
            rewards = []
            validity = []
            
            # questions, solutions = self.separate_qa(queries)
            # gts = [self.q2gt.get(q, None) for q in potential_qids]
            format_type = getattr(self.strategy.args, "format", None)
            sysprompt = getattr(self.strategy.args, "system_prompt", None)
            requires_box = False if self.parse_code or sysprompt in ['dpsk','notrigger'] else True # sysprompt !='autocode'
            print(f'requires_box={requires_box}')
            # num = len(questions)
            
            if use_response and not is_eval: 
                error_infos = [None for _ in range(num)]
                use_codes = [0.0 for _ in range(num)]
                validity = [1.0 for _ in range(num)]
                norepeat_rewards = [1.0 for _ in range(num)]
                usefmt_rewards = [1.0 for _ in range(num)]
                # ns_in_correct = [1.0 for _ in range(num)]
                round0_nwait = [0.0 for _ in range(num)]
                round1_nwait = [0.0 for _ in range(num)]
                raw_rewards = [1.0 for _ in range(num)] 
                exceptions = [0.0 for _ in range(num)]
            else: 
                # rets = round0_correctness 
                # rets = self.rule_reward_func(solutions, gts, self.tokenizer.eos_token, format_type, self.executor, requires_box)
                for iidx,(ret0,ret1) in enumerate(zip(round0_correctness, round1_correctness)):
                    # print('!!!! solution', sol)
                    if ret1 is None: ret = ret0
                    else: ret = ret1
                    if self.parse_code:
                        valid, norepeat, usefmt, error_info, usecode, final_correct = ret
                    else: 
                        valid, norepeat, usefmt, error_info, final_correct = ret
                        usecode = False
                    # tmp = sol.lower()
                    # num_wait = tmp.count('wait,')
                    # num_alter = tmp.count('alternatively')
                    # num_switch = num_wait + num_alter
                    
                    if initial_validity: 
                        valid = initial_validity[iidx] and valid
                    error_infos.append(error_info)
                    use_codes.append(usecode)
                    validity.append(1.0 if valid else 0.0 )
                    norepeat_rewards.append(norepeat) 
                    usefmt_rewards.append(usefmt)
                    # ns_in_correct.append(0.0)
                    raw_rewards.append(1.0 if final_correct>0 else 0.0)
                    exceptions.append(1.0 if final_correct<0 else 0.0)
                    
            # for valid, final_correct in zip(validity, raw_rewards):
            for valid, final_correct, r0nw, r1nw, r1c in zip(validity, raw_rewards, round0_nwait, round1_nwait, round1_correctness):
                if valid>0.5:
                    shaped_reward = 1.0 if final_correct>0.5 else 0.0 
                else:
                    shaped_reward = -0.1
                ########### it seems not proper to use additive rewards
                if not is_eval:
                    # rules:
                    # - round0 corret wait >  round1 self-verification/self-questioning >= round0 correct >  round1 self-correction
                    if r1c is None: # not forced rethinking
                        if final_correct>0.5: # correct and with native wait 
                            if 3>r0nw>0:
                                shaped_reward = 1.1
                            else: # don't want too many waits even if it finally gets correct
                                shaped_reward = max(shaped_reward - r0nw*0.2, -0.1)
                    # else: # forced rethinking 
                    #     if final_correct>0.5:
                    #         if 4>r1nw>0:
                    #             if r1c[-1]<0.5: # self-verification or self-questioning
                    #                 shaped_reward = 1.1
                    #             else: # self-correction
                    #                 shaped_reward = 0.9 
                    #         elif r1nw>3: 
                    #             shaped_reward = max(shaped_reward - r1nw*0.2, -0.1)
                    #     else: # if it is currently incorrect, we may incentivize it to self reflection 
                    #         if 2>r1nw>0:
                    #             shaped_reward = 0.2 
                            
                print(f"===> [verbose] shaped_reward={shaped_reward}, final_correct={final_correct}, r0nw={r0nw}, r1nw={r1nw}, r1c={r1c}")
                ########### not using fmt reward
                # if format_type in ['confidence','nocode']:
                #     if shaped_reward>0.99: 
                #         if usefmt is True: shaped_reward += 0.25 
                #         else: shaped_reward -= 0.25
                #     elif shaped_reward>-0.01: 
                #         if usefmt is False: shaped_reward -= 0.1
                #         else: shaped_reward += 0.1 
                # elif format_type in ['wait']:
                #     if shaped_reward>0.99: 
                #         if usefmt is True: shaped_reward += 1.0
                #         else: shaped_reward -= 0.25
                #     elif shaped_reward>-0.01: 
                #         if usefmt is False: shaped_reward -= 0.1
                #         else: shaped_reward += 0.1 
                ####################################
                rewards.append(shaped_reward)
            print(f"===> [verbose] shaped_reward={rewards}")
            rewards = torch.FloatTensor(rewards) # a list of tensor, tensor shape = queries shape
        # print('!!!! debug rewards', rewards.shape)
        info = {
            "reward": rewards, # tensor of shape (queries)
            "response_length": batched_sample.response_length,
            "total_length": batched_sample.total_length,
            "num_actions": na_each,
            "validity": validity, 
            "norepeat": [0.0 if x is None else float(x) for x in norepeat_rewards],
            "usefmt": [0.0 if x is None else float(x) for x in usefmt_rewards],
            "match": [0.0 if x is None else float(x)  for x in raw_rewards],
            "use_codes": [0.0 if x is None else float(x) for x in use_codes],
            # "num_switch": [float(x) for x in ns_in_correct],
            "round0_nwait": [float(x) for x in round0_nwait],
            "round1_nwait": [float(x) for x in round1_nwait],
            "round0_correctness": [float(x[-1]) for x in round0_correctness],
            "round1_correctness": [-1.0 if x is None else float(x[-1]) for x in round1_correctness],
            "qids": potential_qids,
            # "round0_saturation": batched_sample.round0_saturation,
            "round0_ALLTrue": batched_sample.round0_ALLTrue,
            "round0_ALLFalse": batched_sample.round0_ALLFalse,
            "round0_Easy": batched_sample.round0_Easy,
            "round0_Hard": batched_sample.round0_Hard,
            "round0_Medium": batched_sample.round0_Medium,
        }
            
        if base_action_log_probs is not None:   
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        
        # rewards = [r.to(device) for r in rewards]
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and self.reward_model:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        # log probs
        
        if is_eval or use_response:
            action_log_probs = None
        else:
            print(f"===> [verbose] remoteEMaker make_experience() processing {num_seq} qas for action_logprob")
            action_log_probs = self.actor(sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens,visual_inputs=visual_inputs_cpu)
            action_log_probs = action_log_probs.to('cpu')
        
        if is_eval or use_response:
            kl = None
        elif self.initial_model is not None:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device='cpu')

        if is_eval or use_response:
            kl_mean_log = None 
            kl_mean = None
        else:
            if not self.packing_samples:
                kl_mean = masked_mean(kl.to(action_mask.device), action_mask, dim=-1)
                # print(kl.device, action_mask.device)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                if value is not None:
                    value = unpacking_samples(value, num_actions)

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)
                
            kl_mean_log = kl_mean.detach().cpu().numpy().tolist()
        
        info['kl'] =  kl_mean
        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
            visual_inputs=visual_inputs,
            validity=validity
        )
        
        if self.actor: self.actor.train()  # reset model state
        # print('!!!! [debug] logging on', self.strategy.get_rank())
        if self.strategy.is_rank_0() or is_eval:
            log_file = self.strategy.args.ckpt_path + '/logs'
            import os 
            os.makedirs(log_file, exist_ok=True)
            log_file += '/sample.'
            
            if log_file:
                if is_eval: log_file += f'eval_iter{self.eval_step}_{self.strategy.get_rank()}.jsonl'
                else: log_file += 'jsonl'
                print(f'===> [verbose] actionlogp reward done for batch @rank{self.strategy.get_rank()}, written to log', log_file)
                with open(log_file,'a') as f:
                    
                    dump_info = dict()
                    for k,v in info.items():
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu().numpy().tolist()
                        dump_info[k] = v
                    # print('debug', info['reward'])
                    dump_info['questions'] = questions
                    dump_info['solutions'] = solutions
                    gts = [self.q2gt.get(q, None) for q in dump_info['qids']]
                    dump_info['gts'] = gts 
                    
                    num = len(dump_info['qids']) # 96 
                    # print('!!!! debug ', dump_info)
                    for i in range(num):
                        entry = dict()
                        for k in ['solutions', 'gts', 'round0_correctness', 'round1_correctness','validity', 'reward', 'round1_nwait', 'round0_nwait',  'qids', 'questions', 'num_actions']: # error_info, usefmt, use_codes
                            # if k=='sol': continue 
                            if k not in dump_info: continue 
                            if len(dump_info[k])!=num:
                                raise Exception(f"dump-info key {k}: {len(dump_info[k])} should be {num}")
                            v = dump_info[k][i]
                            
                            entry[k] = v
                        f.write(json.dumps(entry)+'\n')
        del sequences, sequences_cpu, action_log_probs, attention_mask, attention_mask_cpu, visual_inputs, visual_inputs_cpu       
        return experience

    def send_requests_to_vllms(self, rank, all_messages, llms, sampling_params):
        refs = []
        batch_size = (len(all_messages) + len(llms) - 1) // len(llms)
        print(f'!!!! [vllm] {len(all_messages)} messages, bsz_each={batch_size}=nqa, numllm={len(llms)}')
        for i, llm in enumerate(llms):
            messages = all_messages[i * batch_size : (i + 1) * batch_size]
            prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts = self.data_processor.handle_placeholders(prompts)
            
            images = [self.data_processor.get_images_from_messages(m) for m in messages]
            # print('!!!! debug img type', type(images[0][0]))
            vllm_inputs = [{
                        "prompt": p,
                        "multi_modal_data":{"image": imgs} if imgs else dummy_image # a trick to handle text-only queries
                        # "mm_processor_kwargs": {
                        #     "min_pixels": int(os.getenv("MIN_PIXELS", 4 * 28 * 28)),
                        #     "max_pixels": int(os.getenv("MAX_PIXELS", 640 * 28 * 28)),
                        # },
            } for p, imgs in zip(prompts,images)]
            

            refs.append(
                llm.add_requests_vlm.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
            )
        return refs 
    
    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask
    
    def _generate_vllm(self, all_prompts: List[str], use_response=False, skip_generation=False, **kwargs) -> List[Samples]:
        from vllm import SamplingParams
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        maxtoken = kwargs.get("max_new_tokens", 1024) # generate max len
        
        
        args = self.strategy.args
        maxtoken=getattr(args, "max_out_tokens", 2048)
        print(f"!!!! [warning] forcifully using maxtoken={maxtoken} for vllm")
        data_version = getattr(args, "data_version", None)
        do_wait = data_version == "append_wait"
        force_wait = "force_append_wait" in data_version 
        force_eval_wait = "force_append_wait_eval" == data_version
        force_all_wait = data_version == "force_append_wait_all"
        # print(f'!!!! debug replace_wait={do_wait} or {force_wait}')
        do_vlm = getattr(args, 'train_vlm', False)
        stop_tokens = ['<|im_end|>','<|eot_id|>','<|endoftext|>']
        
        print(f'===> [verbose] remoteEMaker _generate_vllm() handling whole batch of {len(all_prompts)} queries')
        skip_generation = use_response
        if use_response:
            
            if not do_vlm:
                questions = [extract_qwen_query_and_response(p)[0] for p in all_prompts]
                raw_qlist = [regularize_text(q) for q in questions] # bug: should use qids instead
                
                sources = [self.tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True).split("<think>")[0] for prompt in all_prompts for _ in range(args.n_samples_per_prompt)]
                targets = [r+self.tokenizer.eos_token for q in raw_qlist for r in self.q2r[q][:args.n_samples_per_prompt]]
                
                assert len(targets)==args.n_samples_per_prompt*len(all_prompts), f"{len(targets)}"
                all_s = self.tokenize_fn(sources, maxtoken, padding=False)["input_ids"] # list of list of ids
                inp_num_tokens = [len(x) for x in all_s]
                out_num_tokens = [maxtoken-x for x in inp_num_tokens]
                
                all_t = [self.tokenize_fn(t, nt, padding=False)["input_ids"] for t,nt in zip(targets, out_num_tokens)] # list of list of ids
                for ttok,nt in zip(all_t, out_num_tokens):
                    print(f'!!!! nt={nt}, realtok={len(ttok)}, valid={ttok[-1]==self.tokenizer.eos_token_id}')
                
                print('!!!! peek targets', [targets[0][:100],'...',targets[0][-100:]])
                
                all_outputs = [(a,b) for a,b in zip(all_s, all_t)]
                print('!!!! num qas', len(all_outputs))
            else:
                all_outputs_offline = []
                all_inputs_offline = []
                for p in all_prompts:
                    chat = json.loads(p) 
                    
                    ###### will be a chat list like this
                    # chat = [dict(role='user', 
                    #          content=[dict(type='image', image=img),
                    #                   dict(type='text', text=q)
                    # ])]
                    # if sysp: chat.insert(0, dict(role='system', content=templates[system_prompt]))
                    # for entry in chat:
                    #     if entry['role']=='user': break
                    qid = chat[-1]['qid']
                    # rq = regularize_text(entry['content'][-1]['text']) 
                    
                    responses = self.q2r[qid][:args.n_samples_per_prompt]
                    # import pdb; pdb.set_trace()
                    cleaned_chat = []
                    for entry in chat:
                        if 'content' in entry:
                            cleaned_chat.append(entry)
                    inputs = self.data_processor(json.dumps(cleaned_chat), self.prompt_max_len, device="cpu")['input_ids'] # output will be a list 
                    
                    # rlist = [rsp+self.tokenizer.eos_token for rsp in responses]
                    for rsp in responses:
                        out = rsp+self.tokenizer.eos_token
                        
                        out_tokens = self.data_processor.processor(
                            text=out,
                            padding=False,
                            max_length=args.generate_max_len,
                            add_special_tokens=False,
                            truncation=True,
                            return_tensors='np',
                        )['input_ids'] # output will be a list 
                        all_outputs_offline.extend(out_tokens)
                        all_inputs_offline.extend(inputs.cpu().numpy().tolist()*len(out_tokens))
                        # print('!!!! [debug]', all_inputs_offline[-1])

        is_eval = kwargs['is_eval']
        
        if is_eval and args.n_samples_per_prompt==1:
            temperature = 0.0 # zero is greedy
            top_p = 1 
            top_k = -1 
        else:
            temperature=getattr(args, "temperature", 1.0)
            top_p=kwargs.get("top_p", 1.0)
            top_k=kwargs.get("top_k", 40)
            if is_eval:
                temperature = getattr(args, "val_temperature", 0.6)
                top_p = 0.95
        
        flag = False 
        all_messages = []
        all_raw_messages = []
        potential_qids = []
        qids_expanded = []
        for m in all_prompts: 
            info = json.loads(m)
            if 'qid' in info[-1]: 
                newm = json.dumps(info[:-1]) # we need to drop the qid entry 
                qid = info[-1]['qid']
                potential_qids.append(qid) 
                qids_expanded.extend([qid]*args.n_samples_per_prompt)
            else: newm = m
            all_messages.extend([newm]*args.n_samples_per_prompt)
            all_raw_messages.extend([m]*args.n_samples_per_prompt)
            
        if is_eval or not skip_generation:
            
            sampling_params = SamplingParams(
                temperature=temperature, 
                top_p=top_p,
                top_k=top_k,
                max_tokens=maxtoken,
                min_tokens=kwargs.get("min_new_tokens", 1),
                skip_special_tokens=kwargs.get("skip_special_tokens", False),
                include_stop_str_in_output=False, # from True to False. 
                stop=stop_tokens,
            )
            print(f'!!!! [vllm] is_eval={is_eval}, sampling args', sampling_params)
            
            refs = []
            rearrange_indices = []
            # if not do_vlm: 
            #     all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
            #     batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
                
            #     # For LLM
            #     all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
            #     for i, llm in enumerate(llms):
            #         prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            #         refs.append(
            #             llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            #         )
            # else: 
                
            batch_size = (len(all_messages) + len(llms) - 1) // len(llms)
            print(f'===> [verbose] to handle {len(all_messages)} qas, bsz={batch_size} qas for {len(llms)} vllm engine.')
            all_vllm_inputs = []
            for i, llm in enumerate(llms):
                messages = all_messages[i * batch_size : (i + 1) * batch_size]
                prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print('!!!! [debug] prompts', [prompts[0]])
                prompts = handle_placeholders(prompts)
                print('!!!! [debug]', [prompts[0]])
                
                images = [self.data_processor.get_images_from_messages(m) for m in messages]
                vllm_inputs = []
                for pp, imgs in zip(prompts, images):
                    tmp = {
                            "prompt": pp,
                            "multi_modal_data":{"image": imgs} # a trick to handle text-only queries  
                    }
                    if imgs is None:
                        raise Exception("images cannot be None")
                    vllm_inputs.append(tmp)
                for x,y,m in zip(prompts, vllm_inputs, messages):
                    mmdata = y["multi_modal_data"]['image']
                    if len(mmdata)!=x.count("<|vision_start|>"):
                        print(f"!!!! debug: {len(mmdata)}; {x.count('<|vision_start|>')}, {m}")
                    
                all_vllm_inputs.extend(vllm_inputs)

                refs.append(
                    llm.add_requests_vlm.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )
            print(f'===> [verbose] {len(all_messages)} QA request submitted to {len(llms)} vllm engine.')
            if flag and rearrange_indices:
                print('!!!! debug rearr', rearrange_indices)
            ray.get(refs)

            # Make sure all requests are sent.
            torch.distributed.barrier()

            # Retrieve and combine results from all outputs
            all_output_refs = []
            for i, llm in enumerate(llms):
                # print('!!!! [vllm] getting responses', i)
                all_output_refs.append(llm.get_responses.remote(rank))
            all_outputs = sum(ray.get(all_output_refs), [])
            if flag and rearrange_indices:
                print('!!!! output fetched', rearrange_indices)
                all_outputs = [all_outputs[i] for i in rearrange_indices]
        
        print(f"===> [verbose] decode and evaluate the initial round of responses")
        all_inputs_ = [list(old_out.prompt_token_ids) for old_out in all_outputs]
        all_outputs_ = [list(old_out.outputs[0].token_ids) for old_out in all_outputs] 
        # all_tokens = [list(out.prompt_token_ids)+list(out.outputs[0].token_ids) for out in group_outs]
        solutions_round0 = all_texts_out = self.tokenizer.batch_decode(all_outputs_, skip_special_tokens=False)
        questions = self.tokenizer.batch_decode(all_inputs_, skip_special_tokens=False)
        questions_cleaned = [x.replace("<|image_pad|>","").replace("<|vision_start|><|vision_end|>","<|vision_start|><|image_pad|><|vision_end|>") for x in questions]
        all_qa_texts = [x+y for x,y in zip(questions_cleaned, solutions_round0)]
        # all_qa_texts_qindex = [len(x) for x,y in zip(questions_cleaned, solutions_round0)]
        # all_texts = [x.replace("<|image_pad|>","").replace("<|vision_start|><|vision_end|>","<|vision_start|><|image_pad|><|vision_end|>") for x in all_texts]
        num_wait_round0 = []
        for x in all_texts_out:
            tmp = x.lower()
            nw1 = tmp.count('wait,')
            nw2 = tmp.count('wait a')
            nalt = tmp.count('alternatively')
            num_wait_round0.append(nw1+nw2+nalt)
        rets_round0 = self.convenient_get_batch_rewards_from_queries(all_qa_texts, qids_expanded)
        print(f"===> [verbose] initial responses acc={np.mean([x[-1] for x in rets_round0])} (total={len(rets_round0)})")
        # print([all_qa_texts[0]])
        print(f"===> [verbose] first generation has {sum(num_wait_round0)}/{len(qids_expanded)} waits per qa.")  
        ####### add saturation calculation here
        nsample = args.n_samples_per_prompt
        difficulty_labels = []
        total = 0
        for idx in range(0, len(rets_round0), nsample):
            group_score = np.mean([x[-1] for x in rets_round0[idx:idx+nsample]])
            if group_score<1./8.:
                difficulty_labels.extend([0]*nsample)
            elif group_score<3./8.: 
                difficulty_labels.extend([1]*nsample)
            elif group_score<6./8.: 
                difficulty_labels.extend([2]*nsample)
            elif group_score<1.: 
                difficulty_labels.extend([3]*nsample)
            else: 
                difficulty_labels.extend([4]*nsample)
            total += 1
        
        #######################################
        ################
        # all_messages -> all_outputs
        # all_raw_messages contains qids 
        ################
        all_inputs_append = []
        all_outputs_append = []
        # do_wait: selective for incorrect samples 
        # force_wait: include wait for correct samples
        rets_composite = [[xx, None] for xx in rets_round0]
        peek_flag = False 
        num_wait_round1 = copy(num_wait_round0)
        solutions_round1 = copy(solutions_round0)
        if  ((force_wait or do_wait) and not is_eval) or (is_eval and (force_eval_wait or force_all_wait) ): 
            # all_outputs_ = [list(old_out.prompt_token_ids) for old_out in all_outputs]
            # all_inputs_ = [list(old_out.outputs[0].token_ids) for old_out in all_outputs] 
            # if force_wait or force_eval_wait: print('!!!! debug force_wait=True')
            print(f"===> [verbose] finding proper qas for a second rethinking")
            req_indexlist = []
            req_alltexts = []
            req_vllminputs = []
            req_qids = []
            req_corr = []
            req_numinputs = []
            num_wait = 0
            for qid in set(potential_qids):
                # the index list of this batch
                selector = [idx for idx,qqid in enumerate(qids_expanded) if qqid==qid]  # the index of this qid
                # print('selctor', selector, 'qids', len(potential_qids), len(qids_expanded))
                group_outs = [all_outputs[idx] for idx in selector]
                group_vllm_inputs = [all_vllm_inputs[idx] for idx in selector]
                out_ids = [out.outputs[0].token_ids for out in group_outs]
                # check whether there is None
                num_none = sum([x is None for x in out_ids])
                if num_none:
                    # print(f'!!!! [vllm] {num_none} None in the batch, skip this batch', [len(x.prompt_token_ids) for x in group_outs])
                    continue
                # all_tokens = [list(out.prompt_token_ids)+list(out.outputs[0].token_ids) for out in group_outs]
                all_texts_this_group = [x for x,qqid in zip(all_qa_texts, qids_expanded) if qid==qqid]
                # all_texts = self.tokenizer.batch_decode(all_tokens, skip_special_tokens=False)
                # all_texts = [x.replace("<|image_pad|>","").replace("<|vision_start|><|vision_end|>","<|vision_start|><|image_pad|><|vision_end|>") for x in all_texts]
                all_numinputs = [len(out.prompt_token_ids) for out in group_outs]
                # num_wait += sum(["wait" in x.lower() or "alternatively" in x.lower() for x in all_texts])
                # rets = self.convenient_get_batch_rewards_from_queries(all_texts, [qid]*len(all_texts))
                rets = [ret for ret,qqid in zip(rets_round0, qids_expanded) if qqid==qid]
         
                match_results = [x[-1] for x in rets] 
                group_score = np.mean(match_results)
                
                if not is_eval: 
                    if do_wait:
                        if group_score>0.6: continue 
                    
                
                selected_positions_in_this_group = []
                one_incorrect = False 
                for ii, (tokens, res) in enumerate(zip(out_ids, match_results)):
                    if tokens[-1]!=self.tokenizer.eos_token_id or len(tokens)>maxtoken-1024: continue 
                    if not is_eval:
                        if force_wait and group_score>0.7 and res<0.5 and not one_incorrect:
                            continue 
                        if (res>0.5 and do_wait): # only include positive ones 
                            continue 
                        if (force_wait and np.random.uniform()<0.35):
                            continue 
                    selected_positions_in_this_group.append(ii)
                    
                    
                selected_positions = selected_positions_in_this_group 
                # print('selected positions', selected_positions, 'max position', len(selector))
                tmp = [selector[iidx] for iidx in selected_positions] # maps to the original index 
                tmp_vllm_inputs = [group_vllm_inputs[iidx] for iidx in selected_positions]
                tmp_texts = [all_texts_this_group[iidx] for iidx in selected_positions] 
                req_indexlist.extend(tmp)
                req_alltexts.extend(tmp_texts) # be careful: this should be full text 
                req_corr.extend([match_results[iidx] for iidx in selected_positions])
                req_numinputs.extend([all_numinputs[iidx] for iidx in selected_positions])
                req_vllminputs.extend(tmp_vllm_inputs)
                req_qids.extend([qid]*len(tmp_vllm_inputs))
                
            
            wait_ratio_round0 = (sum([x>0 for x in num_wait_round0]))/len(qids_expanded)
            # make new requests
            batch_size = (len(req_vllminputs) + len(llms) - 1) // len(llms)
            print(f'===> [vllm] append-wait requests {len(req_vllminputs)} messages, bsz_each={batch_size}=nqa, numllm={len(llms)}')
            triggers = [
                # "\n\nWait, is it correct",
                "\n\nWait",
                "\n\nWait, does it seem right?",
                # "\n\nWait, let's re-examine the image more carefully", # this likely lead to repetition issues
                # "\n\nWait, I need to examine the image more carefully"
                # "\n\nWait, it doesn't seem right. Let's re-examine the image more carefully",
                # "\n\nWait, I'm afraid I made a mistake",
                "\n\nWait, let's double check",
                "\n\nWait, there might be a mistake",
                # "\n\nAlternatively, ",
            ]
            reqs = []
            used_triggers = []
            for i, llm in enumerate(llms):
                vllm_inputs = req_vllminputs[i * batch_size : (i + 1) * batch_size] # dict(prompt, multimodal_data)
                prev_texts = req_alltexts[i * batch_size : (i + 1) * batch_size]
                correctness = req_corr[i * batch_size : (i + 1) * batch_size]
                
                for vinp, pt, ismatch in zip(vllm_inputs, prev_texts, correctness):
                    if pt.endswith(self.tokenizer.eos_token): 
                        npt = pt[:-len(self.tokenizer.eos_token)]
                    else: npt = pt 
                    if np.random.uniform()<0.5:
                        assistant_start = npt.find("<|im_start|>assistant\n")
                        last = npt.rfind("Wait", assistant_start)
                        if last>-1:
                            npt = npt[:last]
                        last_box = npt.rfind("\\boxed", assistant_start)
                        if last_box>-1:
                            npt = npt[:last_box]
                        last_answer = npt.lower().find("answer", assistant_start)
                        if last_answer>-1:
                            npt = npt[:last_answer]
                        last_newline = npt.rfind("\n\n", assistant_start)
                        if last_newline>-1: 
                            npt = npt[:last_newline].strip()
                    
                    if is_eval:
                        selected = 0
                    else:
                        if np.random.uniform()<0.5:
                            selected = 0 if np.random.uniform()<0.4 else 1 
                        else:
                            selected = 2 if ismatch else 3 # 2 if np.random.uniform()<0.5 else 0
                    if not ismatch and do_wait: 
                        selected = np.random.choice(np.arange(len(triggers))) 
                    trigger = triggers[selected]
                    npt = npt+trigger
                    used_triggers.append(selected)
                    
                    vinp['prompt'] = npt # change to concat text for a second try 
                    
                
                tmp_params = copy(sampling_params)
                if not is_eval:
                    tmp_params.temperature = 0.9 
                reqs.append(
                    llm.add_requests_vlm.remote(rank, sampling_params=tmp_params, vllm_vision_input=vllm_inputs)
                )
            print(f"===> [verbose] submit rethinking requests {len(req_indexlist)}")
            # print trigger stats
            trigger_stats = defaultdict(int)
            for idx in used_triggers:
                trigger_stats[idx] += 1
            for k,v in trigger_stats.items():
                print(f"===> [verbose] trigger {triggers[k]} used {v}/{len(used_triggers)} times")
            ray.get(reqs)

            # Make sure all requests are sent.
            torch.distributed.barrier()
            del vllm_inputs, req_vllminputs
            # Retrieve and combine results from all outputs
            new_output_refs = []
            for i, llm in enumerate(llms):
                # print('!!!! [vllm] getting responses', i)
                new_output_refs.append(llm.get_responses.remote(rank))
            new_outputs = sum(ray.get(new_output_refs), [])
            print(f"===> [verbose] decode and evaluate the rethinking responses ")
            all_tokens = [list(out.prompt_token_ids)+list(out.outputs[0].token_ids) for out in new_outputs]
            # all_texts = self.tokenizer.batch_decode(all_tokens, skip_special_tokens=False)
            all_tokens_in = [list(out.prompt_token_ids) for out in new_outputs]
            all_tokens_out = [list(out.outputs[0].token_ids) for out in new_outputs]
            all_texts_in = self.tokenizer.batch_decode(all_tokens_in, skip_special_tokens=False)
            all_texts_out = self.tokenizer.batch_decode(all_tokens_out, skip_special_tokens=False)
            # all_texts = self.tokenizer.batch_decode(all_tokens_out, skip_special_tokens=False)
            
            # rethinking_texts = [triggers[trig]+x for trig, x in zip(used_triggers,all_texts)]
            # print(f"===> peek rethinking responses", rethinking_texts[0])
            # all_texts_inp_short = self.tokenizer.batch_decode([list(out.prompt_token_ids)[:50] for out in new_outputs], skip_special_tokens=False)
            # all_texts = [x.replace("<|image_pad|>","").replace("<|vision_start|><|vision_end|>","<|vision_start|><|image_pad|><|vision_end|>") for x in all_texts]
            all_texts_in = [x.replace("<|image_pad|>","").replace("<|vision_start|><|vision_end|>","<|vision_start|><|image_pad|><|vision_end|>") for x in all_texts_in]
            all_texts = [x+y for x,y in zip(all_texts_in, all_texts_out)]
            # print(f'!!!! [debug] num returned {len(all_texts)}')
            # for tt in all_texts:
            #     print(f"!!!! [debug] peek rethinking response", [tt])
            if len(all_texts)==0:
                rets_round1 = []
            else: 
                # assert len(all_texts)==len(req_qids)
                eg = all_texts[0]
                wait_index = eg.find("Wait")
                if wait_index==-1:
                    wait_index = -100
                print(f"!!!! [debug] peek rethinking response", [eg[:100], '...', eg[wait_index-10:]])
                rets_round1 = self.convenient_get_batch_rewards_from_queries(all_texts, req_qids)
            match_results = [x[-1] for x in rets_round1]
            
            print(f"===> [verbose] rethinking responses acc={np.mean(match_results)}, match_results={match_results}")
            all_inputs_append = [None for _ in range(len(all_outputs))]
            all_outputs_append = [None for _ in range(len(all_outputs))]
            replace_nc = 0
            replace_total = 0
            q2cnt = defaultdict(int)
            drop_reasons = defaultdict(int)
            peek_flag = False
            num_wait_round1_ = []
            
            for out_token, resp, qid, res, oldindex, new_out, num_inputs in zip(all_tokens_out, all_texts_out, req_qids, match_results, req_indexlist, new_outputs, req_numinputs): 
                is_valid = 1024>len(out_token)>30 # there should at least be some outputs
                if not is_valid:
                    drop_reasons['too_short_or_too_long'] += 1 
                    continue 
                ######## only need to show correct rethinking
                if res<0.5:
                    drop_reasons['incorrect'] += 1 
                    continue 
                add_this = False 
                if not is_eval:
                    add_this = force_wait and q2cnt[qid]<2 # no more than three each query
                    # add_this = (res>0.5 or np.random.uniform()<0.3) and force_wait and q2cnt[qid]<2 # no more than three each query
                    if not add_this: add_this = (res>0.5 and do_wait)
                    if not add_this:
                        drop_reasons['many_rethink_for_same_query'] += 1
                        continue 
                    q2cnt[qid] += 1
                else:
                    if force_eval_wait: add_this = True 
                    elif force_all_wait: add_this = res>0.5 # this is upper bound 
                    # if is_valid and add_this: # include both correct and incorrect ones
                ############
                tmp = resp.lower()
                nw1 = tmp.count('wait,')
                nw2 = tmp.count('wait a')
                nalt = tmp.count('alternatively')
                
                num_wait = 1+nw1+nw2+nalt # one wait is included in the inputs 
                
                if num_wait>4: 
                    drop_reasons['too_many_waits']
                    continue # remove repeated wait 
                
                if res>0.5:
                    # print(f"!!!! debug successful {replace_nc} newtext={[qa[:100], '...', qa[]]}")
                    replace_nc += 1
                replace_total += 1
                num_wait_round1_.append(num_wait)
                
                # all_outputs[oldindex] = new_out
                in_token = list(new_out.prompt_token_ids) 
                
                full_token = in_token + out_token 
                # since current out token is generated by prefix continue writing, we should find the original input tokens 
                real_in_token = full_token[:num_inputs]
                real_out_token = full_token[num_inputs:]
                if res<0.5 and len(real_out_token)>1200: # incorrect, avoid long and repeated garbage
                    continue 
                all_inputs_append[oldindex] = real_in_token 
                all_outputs_append[oldindex] = real_out_token 
                
            tmp_idx = 0
            for iidx, old_out in enumerate(all_outputs):
                
                if all_inputs_append[iidx] is not None: 
                    rets_composite[iidx][-1] = rets_round1[tmp_idx]
                    num_wait_round1[iidx] += num_wait_round1_[tmp_idx]
                    solutions_round1[iidx] += all_texts[tmp_idx]
                    tmp_idx += 1
                    continue # these are replaced by append-wait 
                in_token = list(old_out.prompt_token_ids) 
                out_token = list(old_out.outputs[0].token_ids) 
                all_inputs_append[iidx] = in_token 
                all_outputs_append[iidx] = out_token
                
            print(f"===> [replace-wait] nc={replace_nc}/{replace_total}/{len(all_texts)}/{len(rets_round0)}")
            # drop reasons
            print(f"===> [replace-wait] drop reasons")
            for k,v in drop_reasons.items():
                print(f"===> [replace-wait] {k}={v}/{len(all_texts)}")
        
        
        # Warning: We handle either use_response or sampling, but not both cases
        if skip_generation: 
            all_outputs = all_outputs_offline
        samples_list = []
        # groupsize = args.micro_rollout_batch_size
        device = 'cpu'
        groupsize = args.micro_rollout_batch_size
        print(f"===> [verbose] vllm generated {len(all_outputs)} outputs arranged in mrbsz={args.micro_rollout_batch_size}")
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            prompts = all_messages[i : i + self.strategy.args.micro_rollout_batch_size]
            raw_prompts = all_raw_messages[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_wait0 = num_wait_round0[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_wait1 = num_wait_round1[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_correctness = rets_composite[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_q = questions_cleaned[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_s = solutions_round1[i : i + self.strategy.args.micro_rollout_batch_size]
            batch_qids = qids_expanded[i : i + self.strategy.args.micro_rollout_batch_size]
            diff_labels = difficulty_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                if all_inputs_append:
                    in_tokens = all_inputs_append[i : i + self.strategy.args.micro_rollout_batch_size]
                    out_tokens = all_outputs_append[i : i + self.strategy.args.micro_rollout_batch_size]
                elif is_eval or not skip_generation:
                    outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
                    in_tokens = [list(output.prompt_token_ids) for output in outputs]
                    out_tokens = [[404] if output.outputs[0].token_ids is None else list(output.outputs[0].token_ids)  for output in outputs]
                    # print(f"!!!! debug in/out token length", [f"{len(x)}+{len(y)}" for x,y in zip(in_tokens, out_tokens)])
                else:
                    in_tokens = all_inputs_offline[i : i + self.strategy.args.micro_rollout_batch_size]
                    out_tokens = all_outputs_offline[i : i + self.strategy.args.micro_rollout_batch_size]
                    
                max_input_len = max([len(x) for x in in_tokens])
                for output in out_tokens:
                    max_output_len = max(max_output_len, len(output))
                
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for idx, out_token in enumerate(out_tokens):
                    # left padding input
                    input_len = len(in_tokens[idx])
                    input_ids = [pad_token_id] * (max_input_len - input_len) + in_tokens[idx]

                    # right padding output
                    output_len = len(out_token)
                    output_ids = list(out_token) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                if self.actor is None:
                    sequences, attention_mask, action_mask = self.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                else: 
                    sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                
                sequences = sequences.to(device)
                attention_mask = attention_mask.to(device)
                action_mask = action_mask.to(device)
                # Collect for visual input
                visual_inputs = None
                # todo: what if there are mixed image-text and text-only inputs?
                if do_vlm:
                    visual_inputs = self.data_processor(prompts, self.prompt_max_len, device=device)
                    visual_inputs.pop("input_ids")
                    visual_inputs.pop("attention_mask")
                    visual_inputs = {k: v.to(device) for k, v in visual_inputs.items()}
                # print('!!!!! debug using broadcast all prompts', len(broadcast_all_prompts), len(sequences))
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        na_each=[len(x) for x in out_tokens], 
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=raw_prompts,
                        visual_inputs=visual_inputs,
                        round0_nwait=batch_wait0,
                        round1_nwait=batch_wait1, 
                        round0_correctness=[x[0] for x in batch_correctness], # be careful here because each entry is a tuple: valid, norepeat, usefmt, error_info, usecode, final_correct
                        round1_correctness=[x[1] for x in batch_correctness],
                        questions=batch_q,
                        solutions=batch_s,
                        qids=batch_qids,
                        round0_ALLTrue=[float(x==4) for x in diff_labels],
                        round0_Easy=[float(x==3) for x in diff_labels],
                        round0_Medium=[float(x==2) for x in diff_labels],
                        round0_Hard=[float(x==1) for x in diff_labels],
                        round0_ALLFalse=[float(x==0) for x in diff_labels],
                        
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                output_tokens = [list(output.outputs[0].token_ids) for output in outputs]
                for idx, out in enumerate(output_tokens):
                    if out[-1]!=self.tokenizer.eos_token_id:
                        print(f"!!!! shorten to 10 tokens because no eos after maxlen truncation")
                        output_tokens[idx] = out[:10] # trick to save memory by shortening invalid 
                        
                predictions = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
                print('!!!! peek pred', [predictions[0][:100],'...',predictions[0][-100:]])
                
                for i, output in enumerate(outputs):
                    inp_token_ids = output.prompt_token_ids
                    input_len = len(output.prompt_token_ids)
                    out_token_ids = output_tokens[i]
                    output_len = len(out_token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(inp_token_ids + out_token_ids)
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device=device).unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device=device).unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device=device, dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device=device, dtype=torch.float)
                # Collect for visual input
                visual_inputs = None
                if self.data_processor is not None:
                    visual_inputs = self.data_processor(prompts, self.prompt_max_len, device=device)
                    visual_inputs.pop("input_ids")
                    visual_inputs.pop("attention_mask")
                    visual_inputs = {k: v.to(device) for k, v in visual_inputs.items()}
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        # prompts=broadcast_all_prompts,
                        visual_inputs=visual_inputs
                    )
                )
        
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


if __name__ == "__main__":
    sol = "<think>\nOkay, so I need to figure out the correct answer for this question about Geomorph Trading's financials. Let me start by understanding what's being asked. The question is asking for the net working capital and total long-term capital, and then calculate the ratio of debt to total long-term capital.\n\nFirst, I'll look at the image itself. The image shows a simplified balance sheet for Geomorph Trading. The balance sheet has two main sections: Assets and Liabilities. Under Assets, there are Current Assets and Long-term Assets. Current Assets are $100, and Long-term Assets are $500. So, total assets would be Current Assets + Long-term Assets = $100 + $500 = $600.\n\nNext, under Liabilities, there are Current Liabilities and Long-term Liabilities. Current Liabilities are $60, and Long-term Liabilities are $280. So, total liabilities would be Current Liabilities + Long-term Liabilities = $60 + $280 = $340.\n\nNow, Equity is listed as $190. So, total equity is $190.\n\nTo find Net Working Capital, I need to subtract Current Liabilities from Current Assets. That would be Current Assets - Current Liabilities = $100 - $60 = $40.\n\nTotal Long-term Capital is the sum of Long-term Assets and Long-term Liabilities. So, that's $500 + $280 = $780.\n\nThe ratio of debt to total long-term capital is calculated by dividing Long-term Debt by Total Long-term Capital. Long-term Debt is $280, and Total Long-term Capital is $780. So, the ratio is 280/780, which simplifies to approximately 0.36 or 36%.\n\nLooking at the options provided, option (B) has working capital of 110, long-term capital of 540, and a debt ratio of 66%. However, based on the calculations, the correct values are working capital of 40, long-term capital of 780, and a debt ratio of 36%.\n\nTherefore, the correct answer is option (D), which matches the calculated values.\n\nAnswer: D\n</think><|im_end|>"
    gt = "\\boxed{D}"
    print(handle_boxed(sol, gt, "<|im_end|>", "none", requires_box=False))