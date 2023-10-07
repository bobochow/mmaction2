from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip

from mmengine.logging import MMLogger
# from mmaction.utils import get_root_logger
from einops import rearrange

from mmaction.registry import MODELS

# from ..builder import BACKBONES

logger = MMLogger.get_current_instance()

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
                scale=1., num_tadapter=1, num_frames=8, drop_path=0.,
                shift: bool = False, 
                shift_type: str = 'psm'):
        super().__init__()
        self.num_tadapter = num_tadapter
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
        
        self.d_model=d_model
        self.shift = shift
        self.shift_type = shift_type

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if self.shift:
            self.shift_op=PatchShift( inv=False)
            
            # if self.shift_type == 'psm':
            #     # self.shift_op = PatchShift(self.n_head, False, 1)
            #     # self.shift_op_back = PatchShift(self.n_head, True, 1)
            #     self.shift_op = PatchShift(False)
            #     self.shift_op_back = PatchShift(True)
                
            # elif self.shift_type == 'tsm':
            #     self.shift_op = TemporalShift(8)
            


    # def attention(self, x: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    # def cross_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
    #     return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention(
        self, x: torch.Tensor , y: torch.Tensor
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
        
        
        out = aff @ v  # N num_heads Tx D_head
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out) # N Tx D
        
        return out 
    
    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # temporal adaptation
        
        class_token=x[:1,:,:] # 1, BT, D
        
        xt = rearrange(class_token, 'n (b t) d -> t (b n) d', t=self.num_frames)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt),self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=1)
        
        x= torch.cat([x[:1,:,:], xt, x[1:,:,:]], dim=0)
        
        ## prompt tuning
        if self.shift:
            
            # x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            
            xln=self.ln_1(x)
            
            tmp_x=xln[2:, :, :].clone()
            # tmp_x=xln[1:, :, :].clone()
            
            L, NT, C = tmp_x.shape
            T = self.num_frames
            N = NT // self.num_frames
            H = W = int(L**0.5)
            tmp_x = rearrange(tmp_x, '(h w) (b t) c -> b t h w c', b=N, t = T, h=H, w=W, c=C)
            tmp_x = self.shift_op(tmp_x)
            tmp_x = rearrange(tmp_x, 'b t h w c -> (b t) c h w')
            
            tmp_x = tmp_x.view(NT, C, -1).permute(2, 0, 1).contiguous() # P NT C
            tmp_x = torch.cat([xln, tmp_x], dim=0)
            
            # x = x + self.S_Adapter(self.cross_attention(xln,tmp_x,tmp_x))
            x = x + self.S_Adapter(self.attention(xln,tmp_x))
        else:
            ## spatial adaptation
            x = x + self.S_Adapter(self.attention(self.ln_1(x),self.ln_1(x)))
        
        # joint adaptation
        # x=x[:-1,:,:]
        x= torch.cat([x[:1,:,:], x[2:,:,:]], dim=0) # [HW+2, BT, D]
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                num_tadapter=1, scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i], 
        #                                                         shift = True,shift_type = 'tsm' if (i % 2 == 0 and self.shift_type == 'psm') or self.shift_type == 'tsm' else 'psm') for i in range(layers)])
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    scale,
                    num_tadapter,
                    num_frames,
                    dpr[i],
                    shift=True,
                    shift_type='psm',
                )
                for i in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_CLIP_UTUNER(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None ):
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

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate)

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
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name and 'shift_conv' not in name:
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
        super(ViT_CLIP_UTUNER, self).train(mode)
        


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

    # backbone=dict(
    #     type='ViT_CLIP_TPS',
    #     pretrained='openaiclip',
    #     input_resolution=224,
    #     patch_size=16,
    #     num_frames=32,
    #     width=768,
    #     layers=12,
    #     heads=12,
    #     drop_path_rate=0.1),

    input_shape = (3,32,224,224)
    model = ViT_CLIP_UTUNER(pretrained='openaiclip',
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
