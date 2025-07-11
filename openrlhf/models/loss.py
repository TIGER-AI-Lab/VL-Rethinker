# /*
#  * Modified by Haozhe Wang in 2025
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  */

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

def get_print(kl_penalty, tmp):
    return round(masked_mean(kl_penalty, tmp, dim=None).item(), 3)

class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, grpo: bool = False, rloo_sft: bool = False) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.grpo = grpo
        self.rloo_sft = rloo_sft

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        kl_coef: float = 0.0,
        validity: Optional[torch.Tensor] = None,
        raw_rewards=None,
        return_dict=False,
        action_entropy=None
    ) -> torch.Tensor:
        
        # if validity is None: 
        #     val_mask = action_mask 
        # else:
        #     val_mask = validity if action_mask is None else action_mask*validity
        # we will use invalid samples: no-eos/no-boxed
        val_mask = action_mask
        ret = dict()
        
        
        logp_ratio = (log_probs - old_log_probs) * val_mask
        ratio = logp_ratio.exp()
        surr1 = ratio * advantages
        # accordiing to DAPO
        e_low = 0.2
        e_high = 0.35
        clipped_ratio = ratio.clamp(1 - e_low, 1 + e_high)
        surr2 = clipped_ratio * advantages
        loss = torch.max(-surr1, -surr2)
        final_sftloss = 0.0 
        #############
        if self.rloo_sft:
            sftloss = 0. 
            ntokens = 0. 
            for idx,wait in enumerate(raw_rewards):
                if wait>0.99: 
                    adv = advantages[idx][-1]
                    if adv>0.:
                        # loss[idx] = -surr1[idx]
                        # if wait>1.5:
                        #     advantages[idx] = -0.25
                        sftloss += torch.sum(-log_probs[idx]*val_mask[idx]) # for convenience directly use logpratio, because old-logp has no grad 
                        ntokens += torch.sum(val_mask[idx]) 
                        print(f'!!!! [debug] SFT with wait={wait} in {raw_rewards}')
            final_sftloss = sftloss/ntokens if ntokens>0. else 0.0 
            ret['sft_loss'] = final_sftloss
        ####################
        # The k3 estimator is the non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        # kl_penalty = 0.0 
        # logp = log_probs.clamp(-5., 0) # .clamp(-0.6)
        # a = old_log_probs - logp # very large - very small?
        # # logx <= x-1, so x-1-logx >= 0 
        # penalty = a.exp() - 1 - a
        # kl_penalty = penalty.clamp(0., 1.0) 
        ####################
        
        
        final = masked_mean(loss, val_mask, dim=None)
        if final.item()>10:
            valid_surr1 = surr1[action_mask]
            valid_ratio = ratio[action_mask]
            valid_adv = advantages[action_mask]
            min_surr1_index = torch.argmin(valid_surr1)
            corresponding_ratio_element = valid_ratio[min_surr1_index]
            corresponding_advantages_element = valid_adv[min_surr1_index]
            
            print(f"!!!! warning pgloss", final, surr1.min(), corresponding_advantages_element, corresponding_ratio_element)
            
            final = final.clamp(-1.0,1.0)
        
        # if self.rloo_sft:
        #     # now raw_rewards act like some markers 
        #     bool_mask = (torch.FloatTensor(raw_rewards)>0.99).to(float) # at least one wait 
        #     bsz = len(bool_mask)
        #     num_actions = val_mask.size(1)
        #     raw_rewards = bool_mask.unsqueeze(1).expand(bsz, num_actions).to(val_mask.device)
        #     tmp = raw_rewards * val_mask
        #     normalizer = tmp / (tmp.sum()+1e-8)
        #     pos_token_logps = (normalizer * log_probs * raw_rewards).sum() 
        #     ret['sft_loss'] = -pos_token_logps
        
        # pos_sel = (advantages>0).float() * val_mask # (advantages>0).float() * tmp 
        # neg_sel = (advantages<0).float() * val_mask # (advantages<0).float() * tmp 
        # ret['weighted_pos_logp'] = masked_mean(log_probs, pos_sel, dim=None)
        # ret['weighted_neg_logp'] = masked_mean(-log_probs, neg_sel, dim=None)
        # ret['kl_penalty'] = masked_mean(kl_penalty, val_mask, dim=None)
        # if action_entropy is not None: 
        #     allneg = (raw_rewards<0.5).float() * (advantages==0).float() * val_mask 
        #     allneg_entropy = masked_mean(action_entropy, allneg, dim=None)
        #     ret['allneg_entropy'] = allneg_entropy
        # print(f"!!!! [training] pos logp = {get_print(log_probs,pos_sel)}, neg logp = {get_print(log_probs,neg_sel)}, kl={ret['kl_penalty']}, allneg_entropy={ret['allneg_entropy'] if action_entropy is not None else None}")
        ret['actor_loss'] = final 
        
        if return_dict: return ret 
        else: return ret['actor_loss']
        
class SFTLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, grpo: bool = False, rloo_sft: bool = False) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.grpo = grpo
        self.rloo_sft = rloo_sft

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        kl_coef: float = 0.0,
        validity: Optional[torch.Tensor] = None,
        raw_rewards=None,
        return_dict=False,
        action_entropy=None
    ) -> torch.Tensor:
        
        # if validity is None: 
        #     val_mask = action_mask 
        # else:
        #     val_mask = validity if action_mask is None else action_mask*validity
        # we will use invalid samples: no-eos/no-boxed
        val_mask = action_mask
            
        ret = dict()
     
        tmp = (raw_rewards>0.5).float() * val_mask
        normalizer = tmp / (tmp.sum()+1e-8)
        # print('!!!! debug', normalizer.shape, log_probs.shape, raw_rewards.shape)
        pos_token_logps = (normalizer * log_probs * raw_rewards).sum() 
        ret['sft_loss'] = -pos_token_logps
        # ret['kl_penalty'] = masked_mean(kl_penalty, val_mask, dim=None)
        
        
        if return_dict: return ret 
        else: return ret['sft_loss']


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
    
class ScaleBTLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = -F.logsigmoid(chosen_reward - reject_reward) * margin
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
