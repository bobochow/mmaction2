from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
import clip

from mmengine.logging import MMLogger
# from mmaction.utils import get_root_logger
from einops import rearrange

from mmaction.registry import MODELS


logger = MMLogger.get_current_instance()


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

<<<<<<< HEAD
=======
        self.n_segment = 8

>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
    @staticmethod
    def softmax_with_policy(attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N, N
        )
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

<<<<<<< HEAD
    def forward(self, x):

        B, N, C = x.shape

        qkv= self.qkv(x)
=======
    def forward(self, x, policy, sampler):

        B, N, C = x.shape

        qkv= self.qkv(x, policy, sampler)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

<<<<<<< HEAD
        
        attn = attn.softmax(dim=-1)
        
=======
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
<<<<<<< HEAD
        x = self.proj(x)
=======
        x = self.proj(x, policy, sampler)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        x = self.proj_drop(x)
        return x

class AdaptiveTokenSampler(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
<<<<<<< HEAD
        drop_tokens=False
=======
        drop_tokens=False,
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
    ):
        super(AdaptiveTokenSampler, self).__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
<<<<<<< HEAD
        self.out_zero_mask_1 = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.out_zero_mask_2 = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.drop_tokens = drop_tokens
        # self.attn = nn.MultiheadAttention(dim, num_heads)
    
=======
        self.out_zero_mask = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.drop_tokens = drop_tokens

>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
    @staticmethod
    def get_unique_indices(indices: Tensor, max_value: int) -> Tensor:
        """
        :param indices: indices of the tokens to be sampled
        :param max_value: maximum number of the tokens to be sampled
        :return: unique indices of the tokens to be sampled
        """
        sorted_indices = torch.sort(indices, dim=1)[0]

        shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
        unique_indices = torch.where(
            (shift_left - sorted_indices) == 0,
            max_value * torch.ones_like(indices),
            sorted_indices,
        )
        unique_indices = torch.sort(unique_indices, dim=1)[0]
        return unique_indices

    @staticmethod
    def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
        """
        Sample uniformly from y-axis.
        ys tensor contains normalized sampling points that start from 
        ys_start and are evenly distributed in the [0, 1] range along 
        the y-axis for each sample in the batch.
        """

        B = normalized_cdf.shape[0]
        # epsilon = (1 / (n_tokens - 1)) / 2
        ys = (
            torch.linspace(
                start=0,
                end=1.0,
                steps=n_tokens - 1,
                device=normalized_cdf.device,
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        ys_start = (
            torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
            .unsqueeze(-1)
            .expand_as(ys)
        )
        steps = (
<<<<<<< HEAD
            torch.arange(0, n_tokens - 1, device=normalized_cdf.device)
=======
            torch.range(0, n_tokens - 2, device=normalized_cdf.device)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
            .unsqueeze(0)
            .expand_as(ys_start)
        )
        ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

        return ys

    @staticmethod
    def score_assignment_step(attn: Tensor, v: Tensor) -> (Tensor, Tensor):
        """
        Token Score Assignment Step.
        :param attn: attention matrix
        :param v: values
        :return: sorted significance scores and their corresponding indices
        """

        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
<<<<<<< HEAD
        )  # value norm of size [B x L]
        significance_score = attn[:, :, 0].sum(
            dim=1
        )  # attention weights of CLS token of size [B x L]
        significance_score = significance_score * v_norm  # [B x L]
        # significance_score = significance_score[:, 1:]  # [B x L-1]
        # exclude temporal cls token
        significance_score = significance_score[:, 2:]  # [B x L-2]
        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x L-2]
=======
        )  # value norm of size [B x T]
        significance_score = attn[:, :, 0].sum(
            dim=1
        )  # attention weights of CLS token of size [B x T]
        significance_score = significance_score * v_norm  # [B x T]
        significance_score = significance_score[:, 1:]  # [B x T-1]

        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x T-1]
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        sorted_scores, sorted_indices = torch.sort(
            significance_score, descending=False, dim=1
        )

        return sorted_scores, sorted_indices

    def inverse_transform_sampling(
        self,
        sorted_scores: Tensor,
<<<<<<< HEAD
        sorted_indices: Tensor, # BT, L-2
        attn: Tensor,  # BT, H, HW+2, HW+2
        n_tokens: int,
        raw_x: Tensor, # BT, HW+2, D
        
=======
        sorted_indices: Tensor,
        attn: Tensor,
        n_tokens: int,
        raw_x: Tensor,
        n_ref_tokens: int,
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
    ) -> (Tensor, Tensor):
        """
        Sample tokens based on their significance scores.
        """
        B, N, C = raw_x.shape

<<<<<<< HEAD
        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x N-2]
=======
        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8

        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

<<<<<<< HEAD
        # ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
        #     dim=2
        # )  # sampled values from y-axis of size [B, n-2, 1]
        
        ys = self.create_ys(normalized_cdf, n_tokens-1).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n_tokens-2, 1]
        
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 2]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 2))
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
        diff_tokens = ys.shape[1] - (N - 2)
=======
        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n-1, 1]
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 1))
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
        diff_tokens = ys.shape[1] - (N - 1)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        tokens_to_pick_ind = torch.min(
            torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
            dim=2,
        )[
            1
        ]  # [B x n-1]

        # Offsetting token indices
        tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

<<<<<<< HEAD
        # Sort attention matrix and add CLS,T-CLS weights.
        attn_sorted = torch.gather(
            attn[:, :, 2:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, N - 2, N),
        )  # [B x h x T-2 x T]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 2), value=0.0)  # [B x h x T x T]

        # # Sort tokens and add CLS,T-CLS token.
        raw_x_tmp = torch.gather(
            raw_x[:, 2:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 2, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 2), value=0.0)  # [B x n x C]

        unique_indices = self.get_unique_indices(
            indices = tokens_to_pick_ind, max_value = n_tokens - 2
        )[:, : n_tokens - 2] # BT, N-2
=======
        # Sort attention matrix and add CLS weights.
        attn_sorted = torch.gather(
            attn[:, :, 1:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, N - 1, N),
        )  # [B x h x T-1 x T]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x T x T]

        # # Sort tokens and add CLS token.
        raw_x_tmp = torch.gather(
            raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 1, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x n x C]

        unique_indices = self.get_unique_indices(
            indices=tokens_to_pick_ind, max_value=N - 1
        )[:, : N - 1]
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8

        # Prune the attention matrix and input tokens.
        attn_tmp = torch.gather(
            attn_tmp,
            2,
            unique_indices.unsqueeze(1)
            .unsqueeze(3)
<<<<<<< HEAD
            .expand(B, self.num_heads, n_tokens - 2, N),
        )
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 2, C)
        )

        attn_tmp = torch.cat([attn[:, :, 0:2], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:2], raw_x_tmp], dim=1)

        policy = (unique_indices != (N - 2)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 2, 0), value=1.0)
        selected_x = raw_x_tmp
        attn = attn_tmp

        sampler = torch.nonzero(policy) # Z 2

        return selected_x, attn, policy, sampler
    
    def forward(
        self,
        x: Tensor,
        policy: Tensor = None,
        sampler: Tensor = None,
        n_tokens: float = 178,
        raw_x: Tensor = None,
        
        t_cls_token: bool = False,
    ):
        if t_cls_token:
            return super().forward(x) , None , None , None

        
        BT, N, C = x.shape # BT, HW+2, D

        if isinstance(N, Tensor):
            N = N.cpu().item()

        if n_tokens > N:  # Number of tokens to be sampled can't be larger than N.
            n_tokens = N
        if n_tokens <= 1.0:  # When n_tokens is a ratio.
            n_tokens = n_tokens * N
        if n_tokens < 8:  # Number of tokens to be sampled can't be less than 8.
            n_tokens = 8

        n_tokens = round(n_tokens)
        if N < n_tokens:
            n_tokens = N

        qkv = self.qkv(x)
        qkv = qkv.reshape(BT, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        qkv = qkv * policy.unsqueeze(0).unsqueeze(
            2
        )  # Get rid of previously removed tokens.
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x Lq x Lk]

        # --------------------------
        # Token Score Assignment
        # --------------------------

        sorted_scores, sorted_indices = self.score_assignment_step(attn, v)

        # --------------------------
        # Inverse Transform Sampling
        # --------------------------

        selected_x, attn, policy, sampler = self.inverse_transform_sampling(
            sorted_scores, sorted_indices, attn, n_tokens, raw_x
        )

        x = (attn @ v).transpose(1, 2).reshape(BT, attn.shape[2], C) # BT N C

        # Pruning
        if self.drop_tokens:
            # policy BT n_sampled 1
            out_mask_size = policy.sum(1).max().int() # max number of tokens to be sampled

            sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1] # Z 1
            sampler = sampler[:, 0] * n_tokens + sampler[:, 1]
            sampler_input = sampler.unsqueeze(-1).expand(-1, C)
            
            sampler_output = sampler_out.unsqueeze(-1).expand(-1, C)
            flatten_x = x.reshape(-1, C)
            flatten_selected_x = selected_x.reshape(-1, C)

            x_prunned = torch.gather(flatten_x, 0, sampler_input)
            selected_x_prunned = torch.gather(flatten_selected_x, 0, sampler_input)

            out_zero_mask_1 = self.out_zero_mask_1.expand(BT * out_mask_size, -1).type_as(x_prunned)
            out_zero_mask_2 = self.out_zero_mask_2.expand(BT * out_mask_size, -1)
            
            x = out_zero_mask_1.scatter_add(
                0, sampler_output, x_prunned
            ).reshape((BT, out_mask_size, C))
            selected_x = out_zero_mask_2.scatter_add(
                0, sampler_output, selected_x_prunned
            ).reshape((BT, out_mask_size, C))

            policy = (
                out_zero_mask_2[:, 0]
                .scatter(0, sampler_out, 1, reduce="add")
                .reshape(BT, out_mask_size, 1)
            )

        x = self.proj(x)
        x = x * policy
        x = self.proj_drop(x)
        return x, selected_x, policy, sampler
=======
            .expand(B, self.num_heads, n_tokens - 1, N),
        )
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
        )

        attn_tmp = torch.cat([attn[:, :, 0:1], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

        policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
        selected_x = raw_x_tmp
        attn = attn_tmp

        sampler = torch.nonzero(policy)

        return selected_x, attn, policy, sampler
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ATSBlock(nn.Module):
    """
    Transformer Block + ATS
    """

    def __init__(
        self,
        dim,
        num_heads,
<<<<<<< HEAD
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        drop_path=0.0,
        drop_tokens=False,
        scale=1.,
        num_frames=8,
    ):
        super().__init__()
        self.d_model=dim
        self.num_frames = num_frames
        
        self.ln_1 = LayerNorm(self.d_model)
        self.ln_2 = LayerNorm(self.d_model)
        
=======
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        insert_control_point=False,
        drop_tokens=False,
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)

>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        self.attn = AdaptiveTokenSampler(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop_tokens=drop_tokens,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
<<<<<<< HEAD
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.d_model, self.d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.d_model * 4, self.d_model))
        ]))
        
        self.MLP_Adapter = Adapter(self.d_model, skip_connect=False)
        self.S_Adapter = Adapter(self.d_model)
        self.scale = scale
        self.T_Adapter = Adapter(self.d_model, skip_connect=False)
        
=======
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8

    def forward(
        self,
        x,
        n_tokens,
        policy: Tensor = None,
        sampler: Tensor = None,
<<<<<<< HEAD
        
    ):
        ## x shape [ BT, HW+1, D]
        bt, n, d = x.shape
        # temporal adaptation
        class_token=x[:,:1,:] #  BT, 1, D
        
        xt = rearrange(class_token, '(b t) n d -> (b n) t d', t=self.num_frames)
        
        xt = self.T_Adapter(self.attn(self.ln_1(xt),t_cls_token=True)[0])
        
        
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=1)
        # x = x + self.drop_path(xt)
        x= torch.cat([x[:,:1,:], xt, x[:, 1:, :]], dim=1) # [ BT, HW+2, D]
        
        
        x_out, selected_x, policy, sampler = self.attn(
            x=self.ln_1(x),
=======
        n_ref_tokens: int = 197,
    ):
        x_out, selected_x, policy, sampler = self.attn(
            x=self.norm1(x),
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
            policy=policy,
            sampler=sampler,
            n_tokens=n_tokens,
            raw_x=x,
<<<<<<< HEAD
        )
        x = selected_x + self.drop_path(self.S_Adapter(x_out))
        x = x * policy
        x= torch.cat([x[:, :1,:], x[:, 2:,:]], dim=1) # [ BT, HW+1, D]
        
        t_cls_polict=policy[:, 1].unsqueeze(-1)
        policy= torch.cat([policy[:, :1], policy[:, 2:]], dim=1)
        
        xn = self.ln_2(x)
            
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
            
        x = x * policy
        
        policy= torch.cat([policy[:, :1], t_cls_polict ,policy[:, 1:]], dim=1)
        
=======
            n_ref_tokens=n_ref_tokens,
        )
        x = selected_x + self.drop_path(x_out)
        x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        x = x + self.drop_path(out)
        x = x * policy
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        return x, policy

class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                d_model: int,
                n_head: int,
<<<<<<< HEAD
=======
                attn_mask: torch.Tensor = None,
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
                scale=1.,
                num_frames=8,
                drop_path=0.,
                shift: bool = False, 
                shift_type: str = 'psm'):
        super().__init__()
        
        self.d_model=d_model
        
<<<<<<< HEAD
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn=Attention(d_model, n_head, qkv_bias=True)
=======
        self.attn = nn.MultiheadAttention(d_model, n_head)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
<<<<<<< HEAD
=======
        self.attn_mask = attn_mask
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        self.n_head = n_head
        
        self.shift = shift
        self.shift_type = shift_type

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if self.shift:
            self.shift_op=PatchShift( inv=False)

<<<<<<< HEAD
    # def attention(
    #         self, x: torch.Tensor , y: torch.Tensor
    #     ) -> Tuple[torch.Tensor, torch.Tensor]:
            
    #         x=x.permute(1,0,2) #  HW+2, BT, D
    #         y=y.permute(1,0,2) #  HW+2, BT, D
            
    #         q = (x @ self.attn.in_proj_weight[:self.d_model].T
    #             ) + self.attn.in_proj_bias[:self.d_model]

    #         k = (y @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
    #             ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
    #         v = (y @ self.attn.in_proj_weight[-self.d_model:].T
    #             ) + self.attn.in_proj_bias[-self.d_model:]
    #         Tx, Ty, N = q.size(0), k.size(0), q.size(1)
    #         q = q.view(Tx, N, self.attn.num_heads,
    #                 self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
    #         k = k.view(Ty, N, self.attn.num_heads,
    #                 self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
    #         v = v.view(Ty, N, self.attn.num_heads,
    #                 self.attn.head_dim).permute(1, 2, 0, 3)
            
    #         aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)
            
    #         aff = aff.softmax(dim=-1)
            
    #         out = aff @ v  # N num_heads Tx D_head
    #         out = out.permute(2, 0, 1, 3).flatten(2)
    #         out = self.attn.out_proj(out) # N Tx D
            
    #         out = out.permute(1,0,2)
            
    #         return out
=======
    def attention(
            self, x: torch.Tensor , y: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            
            x=x.permute(1,0,2) #  HW+2, BT, D
            y=y.permute(1,0,2) #  HW+2, BT, D
            
            q = (x @ self.attn.in_proj_weight[:self.d_model].T
                ) + self.attn.in_proj_bias[:self.d_model]

            k = (y @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
                ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
            v = (y @ self.attn.in_proj_weight[-self.d_model:].T
                ) + self.attn.in_proj_bias[-self.d_model:]
            Tx, Ty, N = q.size(0), k.size(0), q.size(1)
            q = q.view(Tx, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
            k = k.view(Ty, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
            v = v.view(Ty, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3)
            
            aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)
            
            aff = aff.softmax(dim=-1)
            
            out = aff @ v  # N num_heads Tx D_head
            out = out.permute(2, 0, 1, 3).flatten(2)
            out = self.attn.out_proj(out) # N Tx D
            
            out=out.permute(1,0,2)
            
            return out
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8

    def forward(
        self,
        x: torch.Tensor
    ):
        ## x shape [ BT, HW+1, D]
        bt, n, d = x.shape
        # temporal adaptation
        class_token=x[:,:1,:] #  BT, 1, D
        
        xt = rearrange(class_token, '(b t) n d -> (b n) t d', t=self.num_frames)
        x_ln1=self.ln_1(xt)
<<<<<<< HEAD
        xt = self.T_Adapter(self.attn(x_ln1))
=======
        xt = self.T_Adapter(self.attention(x_ln1,x_ln1)[0])
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=1)
        # x = x + self.drop_path(xt)
        x= torch.cat([x[:,:1,:], xt, x[:, 1:, :]], dim=1) # [ BT, HW+2, D]
        
        # spatial adaptation
        x_ln2=self.ln_1(x)
<<<<<<< HEAD
        x = x + self.S_Adapter(self.attn(x_ln2))
=======
        x = x + self.S_Adapter(self.attention(x_ln2,x_ln2))
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        
        x= torch.cat([x[:, :1,:], x[:, 2:,:]], dim=1) # [ BT, HW+1, D]
        xn = self.ln_2(x)
        
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        
        return x


class Transformer(nn.Module):
<<<<<<< HEAD
    def __init__(self, num_frames, 
                width: int,
                layers: int,
                heads: int,
                scale=1.,
                drop_path=0.1,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                drop_tokens=True,
                ats_blocks=None,
                num_tokens=None):
=======
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                scale=1., drop_path=0.1,ats_blocks=None,num_tokens=None):
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        super().__init__()
        self.width = width
        self.layers = layers

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        
        self.ats_blocks = ats_blocks
        self.num_tokens = num_tokens
        
        # self.resblocks = nn.Sequential(
        #     *[
        #         ResidualAttentionBlock(
        #             width,
        #             heads,
        #             attn_mask,
        #             scale,
        #             num_frames,
        #             dpr[i],
        #             shift=True,
        #             shift_type='psm',
        #         )
        #         for i in range(layers)
        #     ]
        # )
        
<<<<<<< HEAD
        self.resblocks = []
        for i in range(layers):
            if i in self.ats_blocks:
                self.resblocks.append(
                    ATSBlock(
                        dim=width,
                        num_heads=heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        drop_path=dpr[i],
                        drop_tokens=drop_tokens,
                        scale=scale,
                        num_frames=num_frames,
                    )
                )
            else:
                self.resblocks.append(
                    ResidualAttentionBlock(
                    width,
                    heads,
=======
        self.blocks = []
        for i in range(layers):
            if i in self.ats_blocks:
                self.blocks.append(
                    ATSBlock(
                        dim=width,
                        num_heads=heads,
                        drop_path=dpr[i],
                        
                    )
                )
            else:
                self.blocks.append(
                    ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
                    scale,
                    num_frames,
                    dpr[i],
                    shift=False,
                    shift_type='psm',
                    )
                )
<<<<<<< HEAD
        self.resblocks = nn.ModuleList(self.resblocks)
=======
        self.blocks = nn.ModuleList(self.blocks)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        
        
    def forward(self, x: torch.Tensor):
        
        B = x.shape[0]
<<<<<<< HEAD
        init_n = x.shape[1]+1
=======
        init_n = x.shape[1]
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        # policies = []
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        sampler = torch.nonzero(policy)
        # idx = 0
<<<<<<< HEAD
        for idx, blk in enumerate(self.resblocks):
=======
        for idx, blk in enumerate(self.blocks):
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
            if idx in self.ats_blocks:
                x, policy = blk(
                    x=x,
                    n_tokens=self.num_tokens[idx],
                    policy=policy,
                    sampler=sampler,
<<<<<<< HEAD
=======
                    n_ref_tokens=init_n,
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
                )
                # idx += 1
                # policies.append(policy)
            else:
                x = blk(x)
                # policies.append(policy)
        
        return x


@MODELS.register_module()
class ViT_CLIP_ATS(nn.Module):
    ## ViT definition in CLIP image encoder
<<<<<<< HEAD
    def __init__(self, 
                input_resolution: int, 
                num_frames: int, 
                patch_size: int, 
                width: int, 
                layers: int, 
                heads: int,
                drop_path_rate,  
                adapter_scale=0.5, 
                pretrained=None,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                drop_tokens=True,
                # ats_blocks=[3, 4, 5, 6, 7, 8, 9, 10, 11],
                # ats_blocks=[8, 9, 10, 11],
                ats_blocks=[6],
                num_tokens=[198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198]):
=======
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate,  adapter_scale=0.5, pretrained=None,
                ats_blocks=[3, 4, 5, 6, 7, 8, 9, 10, 11],
                num_tokens=[197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197]):
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        
<<<<<<< HEAD
        self.transformer = Transformer( num_frames, 
                                        width, 
                                        layers, 
                                        heads,  
                                        scale=adapter_scale, 
                                        drop_path=drop_path_rate,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop=attn_drop,
                                        drop_tokens=drop_tokens,
                                        ats_blocks=ats_blocks,
                                        num_tokens=num_tokens)
=======
        self.transformer = Transformer(num_frames, width, layers, heads,  scale=adapter_scale, drop_path=drop_path_rate,ats_blocks=ats_blocks,num_tokens=num_tokens)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8

        self.ln_post = LayerNorm(width)
        
        self.ats_blocks = ats_blocks
        self.num_tokens = num_tokens


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
<<<<<<< HEAD
            
            swaps = [('in_proj_', 'qkv.'), ('out_proj', 'proj')]
            
            out_dict={}
            for k, v in pretrain_dict.items():
                for sp in swaps:
                    k = k.replace(sp[0], sp[1])
            
                out_dict[k] = v
            
            msg = self.load_state_dict(out_dict, strict=False)
            
            # msg = self.load_state_dict(pretrain_dict, strict=False)
=======
            msg = self.load_state_dict(pretrain_dict, strict=False)
>>>>>>> 3189cb338d76331c77ebb96f78980b8d2bf557f8
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            logger.info('No pretrained weights are loaded!!!!!')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        self._freeze_stages()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) # bt d n
        x = x.permute(0, 2, 1)  # bt n d
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND    [HW+1, NT, D]
        
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
    
    def _freeze_stages(self) -> None:
        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')
        
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info(
            f'Number of total parameters: {(num_total_param/1.e6):6.2f}, tunable parameters: {(num_param/1.e6):6.2f}'
        )
        
    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(ViT_CLIP_ATS, self).train(mode)
        


class PatchShift(nn.Module):
    def __init__(self, inv=False, ratio=1):
        super(PatchShift, self).__init__()

        self.inv = inv

        if inv:
            print('=> Using inverse PatchShift,   tps')
        else:
            print('=> Using PatchShift,   tps')

    def forward(self, x):
        x = self.shift(x, inv=self.inv)
        return x #self.net(x)

    @staticmethod
    def shift(x, inv=False):
        B, T , H, W, c = x.size()
        feat = x
        # feat = feat.view(B, T,  H,  W, c)
        out = feat.clone()
        stride = 1
        multiplier = -1 if inv else 1
        ## Pattern C
        out[:, :, 0::3, 0::3,:] = torch.roll(feat[:, :,  0::3,0::3,:], shifts=-4*multiplier*stride, dims=1)
        out[:, :, 0::3, 1::3,:] = torch.roll(feat[:, :,  0::3,1::3,:], shifts=multiplier*stride, dims=1)
        out[:, :, 1::3, 0::3,:] = torch.roll(feat[:, :,  1::3,0::3,:], shifts=-multiplier*stride, dims=1)
        out[:, :, 0::3, 2::3,:] = torch.roll(feat[:, :,  0::3,2::3,:], shifts=2*multiplier*stride, dims=1)
        out[:, :, 2::3, 0::3,:] = torch.roll(feat[:, :,  2::3,0::3,:], shifts=-2*multiplier*stride, dims=1)
        out[:, :, 1::3, 2::3,:] = torch.roll(feat[:, :,  1::3,2::3,:], shifts=3*multiplier*stride, dims=1)
        out[:, :, 2::3, 1::3,:] = torch.roll(feat[:, :,  2::3,1::3,:], shifts=-3*multiplier*stride, dims=1)
        out[:, :, 2::3, 2::3,:] = torch.roll(feat[:, :,  2::3,2::3,:], shifts=4*multiplier*stride, dims=1) 

        # out = out.view(B, T, H, W, c)
        return out



if __name__ == '__main__':

    from mmengine.analysis import get_model_complexity_info

    input_shape = (3,32,224,224)
    model = ViT_CLIP_ATS(pretrained='openaiclip',
                        input_resolution=224,
                        adapter_scale=0.5,
                        patch_size=16,
                        num_frames=32,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        froze=True)

    analysis_results = get_model_complexity_info(model, input_shape)


    print(analysis_results['out_table'])

    # print(analysis_results['out_arch'])

    print(f"Model Flops:{analysis_results['flops_str']}")

    print(f"Model Parameters:{analysis_results['params_str']}")
