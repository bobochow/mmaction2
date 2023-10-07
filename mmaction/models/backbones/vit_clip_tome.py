from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip

from typing import List, Tuple, Union

from mmengine.logging import MMLogger
# from mmaction.utils import get_root_logger
from einops import rearrange

from mmaction.registry import MODELS

from timm.models.vision_transformer import Attention

import math
from typing import Callable, Tuple

logger = MMLogger.get_current_instance()

def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    return int(number) == number

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # BT HW+1 D/head(64)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2) # BT HW/2+1 HW/2

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # BT HW/2+1 1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]
        

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    
    if size is None:
        size = torch.ones_like(x[..., 0, None]) # (N, L, 1)

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        # BT, HW+2, D
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        
        # Return k as well here 
        return x, k.mean(1)

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
    def __init__(self, d_model: int, n_head: int, _tome_info: dict,attn_mask: torch.Tensor = None,
                scale=1., num_frames=8, drop_path=0.,
                shift: bool = False, 
                shift_type: str = 'psm'):
        super().__init__()
        
        self._tome_info=_tome_info
        self.d_model = d_model
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.attn =ToMeAttention(d_model, n_head,qkv_bias=True)
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

    # def attention(self, x: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    # def cross_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
    #     return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def attention(
        self, x: torch.Tensor ,size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # input :  BT, HW+2, D
        x=x.permute(1,0,2) #  HW+2, BT, D
        
        q = (x @ self.attn.in_proj_weight[:self.d_model].T
             ) + self.attn.in_proj_bias[:self.d_model]

        k = (x @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
             ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
        v = (x @ self.attn.in_proj_weight[-self.d_model:].T
             ) + self.attn.in_proj_bias[-self.d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)

        # Apply proportional attention
        if size is not None:
            aff = aff + size.log()[:, None, None, :, 0] # (N, 1, 1, Tx)
        
        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        
        out=out.permute(1,0,2)

        # Return k as well here
        return out, k.mean(1)
    
    def cross_attention(
        self, x: torch.Tensor, y: torch.Tensor, size: torch.Tensor = None
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
                   self.attn.head_dim).permute(1, 2, 0, 3)       # N num_heads Tx D_head
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)       # N num_heads Ty D_head
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)

        # Apply proportional attention
        if size is not None:
            aff[:,:,:,:Tx] = aff[:,:,:,:Tx] + size.log()[:, None, None, :, 0]
        
        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        
        out=out.permute(1,0,2)

        # Return k as well here
        return out, k.mean(1)[:,:Tx,:]
    

    def forward(self, x: torch.Tensor):
        ## x shape [ BT, HW+1, D]
        bt, n, d = x.shape
        # temporal adaptation
        class_token=x[:,:1,:] #  BT, 1, D
        
        xt = rearrange(class_token, '(b t) n d -> (b n) t d', t=self.num_frames)
        
        xt = self.T_Adapter(self.attention(self.ln_1(xt),None)[0])
        # xt = self.T_Adapter(self.attn(self.ln_1(xt),None)[0])
        
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=1)
        # x = x + self.drop_path(xt)
        x= torch.cat([x[:,:1,:], xt, x[:, 1:, :]], dim=1) # [ BT, HW+2, D]
        
        ## prompt tuning
        if self.shift:
            xln=self.ln_1(x)
            tmp_x=xln[:, 2:, :].clone()
            
            NT, L, C = tmp_x.shape
            T = self.num_frames
            N = NT // self.num_frames
            # H = W = int(L**0.5)
            H,W=crack(L)
            tmp_x = rearrange(tmp_x, ' (b t) (h w) c -> b t h w c', b=N, t = T, h=H, w=W, c=C)
            tmp_x = self.shift_op(tmp_x)
            tmp_x = rearrange(tmp_x, 'b t h w c -> (b t) c h w')
            
            tmp_x = tmp_x.view(NT, C, -1).permute(0, 2, 1).contiguous() # NT L C
            tmp_x = torch.cat([xln, tmp_x], dim=1)
            
            
            
            attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            x_attn, metric = self.cross_attention(xln, tmp_x ,attn_size)
            
            x = x + self.drop_path(self.S_Adapter(x_attn))
            
            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                    
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

            
            x= torch.cat([x[:,:1,:], x[:,2:,:]], dim=1) # [HW+2, BT, D]
            
            x = x + self.mlp(self.ln_2(x)) + self.drop_path(self.scale * self.MLP_Adapter(self.ln_2(x)))
            
            # x = x + self.drop_path(self.S_Adapter(self.cross_attention(xln,tmp_x,tmp_x)))
        else:
            ## spatial adaptation
            
            # x= x.permute(1,0,2) # BT, HW+2, D
            
            attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            # x_attn, metric = self.attn(self.ln_1(x), attn_size)
            x_attn, metric = self.attention(self.ln_1(x), attn_size)
            
            x = x + self.drop_path(self.S_Adapter(x_attn))
            
            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                    
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

            
            x= torch.cat([x[:, :1,:], x[:, 2:,:]], dim=1) # [ BT, HW+1, D]
            xn = self.ln_2(x)
            
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
            
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, _tome_info: dict,attn_mask: torch.Tensor = None, 
                scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        self._tome_info=_tome_info
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i], 
        #                                                         shift = True,shift_type = 'tsm' if (i % 2 == 0 and self.shift_type == 'psm') or self.shift_type == 'tsm' else 'psm') for i in range(layers)])
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    self._tome_info,
                    attn_mask,
                    scale,
                    num_frames,
                    dpr[i],
                    shift= i in {9,6,3},
                    shift_type='psm',
                )
                for i in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_CLIP_TOME(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, adapter_scale=0.5, pretrained=None ,
                trace_source: bool = False, prop_attn: bool = True, tome_r: Tuple[int, float]= (8,0)):
        
        """
        Applies ToMe to this transformer. Afterward, set r using model.r.

        If you want to know the source of each token (e.g., for visualization), set trace_source = true.
        The sources will be available at model._tome_info["source"] afterward.

        For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
        the shelf. For trianing and for evaluating MAE models off the self set this to be False.
        """
        
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
        self.tome_r=tome_r
        self._tome_info = {
        "r": tome_r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": True,
        }
        
        

        self.transformer = Transformer(num_frames, width, layers, heads, self._tome_info, scale=adapter_scale, drop_path=drop_path_rate )

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
            
            # swaps = [('in_proj_', 'qkv.'), ('out_proj', 'proj')]
            
            # out_dict={}
            # for k, v in pretrain_dict.items():
            #     for sp in swaps:
            #         k = k.replace(sp[0], sp[1])
                    
            #     out_dict[k] = v
            
            # msg = self.load_state_dict(out_dict, strict=False)
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
        
        self._tome_info["r"] = parse_r(self.layers, self.tome_r)
        self._tome_info["size"]=None
        
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
            if 'temporal_embedding' not in name and 'ln_post' not in name  and 'Adapter' not in name :
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
        super(ViT_CLIP_TOME, self).train(mode)
        


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
    model = ViT_CLIP_TOME(pretrained='openaiclip',
                        input_resolution=224,
                        adapter_scale=0.5,
                        patch_size=16,
                        num_frames=32,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        )

    analysis_results = get_model_complexity_info(model, input_shape)


    print(analysis_results['out_table'])

    # print(analysis_results['out_arch'])

    print(f"Model Flops:{analysis_results['flops_str']}")

    print(f"Model Parameters:{analysis_results['params_str']}")
