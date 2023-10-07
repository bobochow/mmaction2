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

def expand_row_index(index, target_shape):
    old_shape = index.shape
    new_dims = len(target_shape) - index.ndim
    
    index = index.view(old_shape[:-1] + (1,) * (new_dims - 1) + (old_shape[-1], 1))
    index = index.expand(target_shape[:-2] + (-1, target_shape[-1]))
    return index

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

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                scale=1., num_frames=8, drop_path=0., t_ats_fraction=None,s_ats_fraction=None,
                shift: bool = False, 
                shift_type: str = 'psm'):
        super().__init__()
        self.t_ats_fraction=t_ats_fraction
        self.s_ats_fraction=s_ats_fraction
        self.d_model=d_model
        
        # self.pre_s_ats_indices = None
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
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
        
    # A simple version of the method from
    # "Adaptive Token Sampling for Efficient Vision Transformers"
    # (Fayyaz et al., ECCV 2022)
    # For now we just use the top-k version of ATS (select the tokens
    # with the k highest scores). Using CDF-based token sampling should
    # also be possible, but it would be more complex to implement (we
    # would need a mechanism for masking the K' < K active tokens in
    # gates and buffers).
    def _adaptive_token_sampling(self, a, v, ats_fraction):
        if ats_fraction is None:
            return a, None

        # a (N, num_heads, Tx, Ty)
        # v (N, num_heads, Ty, D_head)
        
        class_scores = a[..., 0].clone()
        
        raw_scores = class_scores * torch.linalg.vector_norm(v[...], dim=-1)
        scores = raw_scores / raw_scores[..., 1:].sum(dim=-1, keepdim=True)

        # Always select the class token.
        scores[..., 0] = float("-inf")
        
        scores[..., 1] = float("-inf")

        # Sum scores over heads.
        scores = scores.sum(dim=1)

        # Add +1 for the class token and +1 for the t_cls token
        
        n_select = int(ats_fraction * (scores.shape[-1] - 1)) + 1
        
        # Select the k tokens with the highest scores.
        ats_indices = scores.topk(n_select, sorted=False)[1]

        # Sort the token indices (for stabilization). This seems to
        # work pretty well, although we could probably come up with
        # better/more sophisticated. E.g., we could try to find the
        # permutation of indices that minimized some norm between the
        # previous and current ats_indices.
        
        # ats_indices = self._stabilize_ats_indices(ats_indices)
        # self.last_ats_indices = ats_indices

        # return (
        #     a.gather(dim=-2, index=expand_row_index(ats_indices, a.shape)),
        #     ats_indices,
        # )
        return ats_indices
    
    def _adaptive_frame_sampling(self, a, v, ats_fraction):
        if ats_fraction is None:
            return a, None

        # a (N, num_heads, Tx, Ty)
        # v (N, num_heads, Ty, D_head)
        
        
        scores=a.clone()
        
        for i in range(scores.shape[-1]):
            raw_scores = scores[...,i] * torch.linalg.vector_norm(v[...], dim=-1) 
            scores[...,i] = raw_scores / raw_scores.sum(dim=-1, keepdim=True)

        # Always don't select the class token of current frame.
        for i in range(scores.shape[-1]):
            scores[:,:,i,i] = float("-inf")
        
        # Sum scores over heads.
        scores = scores.sum(dim=1) # N, Tx, Ty

        # Add +1 for the class token
        
        n_select = int(ats_fraction)
        
        # Select the k tokens with the highest scores.
        ats_indices = scores.topk(n_select, dim=-1,sorted=False)[1]

        # Sort the token indices (for stabilization). This seems to
        # work pretty well, although we could probably come up with
        # better/more sophisticated. E.g., we could try to find the
        # permutation of indices that minimized some norm between the
        # previous and current ats_indices.
        
        # ats_indices = self._stabilize_ats_indices(ats_indices)
        # self.last_ats_indices = ats_indices

        # return (
        #     a.gather(dim=-2, index=expand_row_index(ats_indices, a.shape)),
        #     ats_indices,
        # )
        return ats_indices
    
    def _stabilize_ats_indices(self, ats_indices):
        
        # ats_indices = scores.topk(n_select, dim=-1,sorted=False)[1]
        # ats_indices: N, k
        
        ats_indices = ats_indices.sort(dim=-1)[0]
        if self.last_ats_indices is None:
            return ats_indices

        # Faster on the CPU
        new_indices = ats_indices.flatten(end_dim=-2).cpu()
        old_indices = self.last_ats_indices.flatten(end_dim=-2).cpu()
        stabilized = old_indices.clone()
        for i in range(new_indices.shape[0]):
            old_not_in_new = torch.isin(old_indices[i], new_indices[i], invert=True)
            new_not_in_old = torch.isin(new_indices[i], old_indices[i], invert=True)
            stabilized[i, old_not_in_new] = new_indices[i, new_not_in_old]
        return stabilized.to(ats_indices.device).view(ats_indices.shape)
    
    @staticmethod
    def _gather_ats_skip(skip_1, ats_indices):
        if ats_indices is None:
            return skip_1
        else:
            return skip_1.gather(
                dim=0, index=expand_row_index(ats_indices, skip_1.shape)
            )
    
    def attention(
        self, x: torch.Tensor , y: torch.Tensor,ats_fraction=None,t_msa=False,c_msa=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # input: HW+2, BT, D
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
        
        ats_indices=None
        with torch.no_grad():
            if t_msa:
                # aff: N, num_heads, Tx, Ty
                # ats_indices: N, Tx , Kt
                ats_indices = self._adaptive_frame_sampling(aff, v, ats_fraction)
            
            elif c_msa:
                # aff: NT, num_heads, Tx, Ty
                # ats_indices: NT, Ks 
                ats_indices = self._adaptive_token_sampling(aff[:,:,:,:Tx], v[:,:,:Tx,:], ats_fraction)
            
            else:
                # aff: NT, num_heads, Tx, Tx
                # ats_indices: NT, Ks
                ats_indices = self._adaptive_token_sampling(aff, v, ats_fraction)
        
        
        out = aff @ v  # N num_heads Tx D_head
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out) # N Tx D
        
        return out , ats_indices
    
    def forward(self, x: torch.Tensor, pre_s_ats_indices=None):
        
        
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal adaptation
        
        class_token=x[:1,:,:] # 1, BT, D
        
        xt = rearrange(class_token, 'n (b t) d -> t (b n) d', t=self.num_frames)
        xt,t_ats_indices=self.attention(self.ln_1(xt), self.ln_1(xt),self.t_ats_fraction,t_msa=True)
        
        xt = self.T_Adapter(xt)
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=1)
        
        x= torch.cat([x[:1, :, :], xt, x[1:, :, :]], dim=0)
        
        ## prompt tuning
        if self.shift and pre_s_ats_indices is not None:
            xln=self.ln_1(x)
            tmp_x=xln.clone() 
            L, NT, C = tmp_x.shape
            T = self.num_frames
            N = NT // self.num_frames
            # H = W = int(L**0.5)
            t_ats_indices=t_ats_indices.permute(1,0,2)
            
            tmp_x = rearrange(tmp_x, 'L (b t) c -> (b t) L c', b = N, t = T, L = L, c = C)
            tmp_x=tmp_x.gather(dim=-2, index=expand_row_index(pre_s_ats_indices, tmp_x.shape))
            tmp_x = rearrange(tmp_x, '(b t) L c -> b t (L c)', b = N, t = T, c = C)
            
            shift_list =[]
            
            for i in range(T):
                shift_token=tmp_x.gather(dim=-2, index=expand_row_index(t_ats_indices[i], tmp_x.shape))
                shift_token=rearrange(shift_token, 'b t (L c) -> b (t L) c', b = N, c = C).permute(1,0,2)
                shift_list .append(shift_token)
            
            tmp_c=torch.stack(shift_list, dim=0)
            tmp_c=rearrange(tmp_c, 't l b c -> l (b t) c', b = N, t = T, c = C)
            tmp_c = torch.cat([xln, tmp_c], dim=0)
            
            x_shift_attn,s_ats_indices=self.attention(xln,tmp_c,self.s_ats_fraction,c_msa=True)
            x = x + self.S_Adapter(x_shift_attn)
        else:
            ## spatial adaptation
            
            x_attn, s_ats_indices = self.attention(self.ln_1(x),self.ln_1(x),self.s_ats_fraction)
            # x_attn=self.attention(self.ln_1(x),self.ln_1(x))
            x_attn = self.S_Adapter(x_attn)
            
            x= x + x_attn
            
        ## joint adaptation
        x= torch.cat([x[:1,:,:], x[2:,:,:]], dim=0)
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x ,s_ats_indices


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                scale=1., drop_path=0.1,t_ats_fraction=None,s_ats_fraction=None):
        super().__init__()
        self.width = width
        self.layers = layers

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    scale,
                    num_frames,
                    dpr[i],
                    t_ats_fraction,
                    s_ats_fraction,
                    shift=True,
                    shift_type='psm',
                )
                for i in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        
        s_indices=None
        
        for blk in self.resblocks:
            x,s_indices = blk(x,s_indices)
        
        return x


@MODELS.register_module()
class ViT_CLIP_ATS_TOPK(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate,  adapter_scale=0.5, pretrained=None,
                t_ats_fraction=None,s_ats_fraction=None):
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

        self.t_ats_fraction=t_ats_fraction
        self.s_ats_fraction=s_ats_fraction
        
        self.transformer = Transformer(num_frames, width, layers, heads,  scale=adapter_scale, drop_path=drop_path_rate,t_ats_fraction=t_ats_fraction,s_ats_fraction=s_ats_fraction)

        self.ln_post = LayerNorm(width)
        
        

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
            msg = self.load_state_dict(pretrain_dict, strict=False)
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

        x = x.permute(1, 0, 2)  # NLD -> LND    [HW+1, NT, D]
        
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
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
        super(ViT_CLIP_ATS_TOPK, self).train(mode)
        


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
    model = ViT_CLIP_ATS_TOPK(pretrained='openaiclip',
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
