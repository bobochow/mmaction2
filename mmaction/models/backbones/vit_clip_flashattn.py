from collections import OrderedDict
from typing import Tuple, Union,Optional
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
from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp

from torch.utils.checkpoint import checkpoint

logger = MMLogger.get_current_instance()

# util functions to convert OpenCLIP-style model keys to ViT-style
def remap_keys_from_open_clip_to_vit(
    clip_state_dict,
    visual_transformer_layers=12,
    textual_transformer_layers=12,
    context_length=77,
    vocab_size=49408,
    use_fast_conv1=False,
    use_flash_attn=False,
):
    if 'state_dict' in clip_state_dict:
        clip_state_dict = clip_state_dict['state_dict']
    if list(clip_state_dict.keys())[0].startswith('module.'):
        clip_state_dict = OrderedDict({
            k.replace('module.', ''): v for k, v in clip_state_dict.items()
        })
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "logit_scale": "logit_scale",
        "visual.proj": "visual.image_projection",
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.text_projection",
        "token_embedding.weight": "textual.token_embedding.weight",
        "ln_final.weight": "textual.ln_final.weight",
        "ln_final.bias": "textual.ln_final.bias"
    }

    for layer in range(visual_transformer_layers):
        if use_flash_attn:
            for src_name, tgt_name in {
                'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
                'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
                'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
            }.items():
                key_mapping[f"visual.transformer.resblocks.{layer}.{src_name}"] = f"visual.transformer.resblocks.{layer}.{tgt_name}"


    for layer in range(textual_transformer_layers):
        for name in [
            'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
            'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
             'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias',
        ]:
            key_mapping[f"transformer.resblocks.{layer}.{name}"] = f"textual.transformer.resblocks.{layer}.{name}"

    for key in clip_state_dict:
        if key in ["visual.proj", "text_projection", "logit_scale"]:
            continue
        if use_fast_conv1 and key == 'visual.conv1.weight':
            remapped_state_dict['visual.conv1.weight'] = clip_state_dict[key].flatten(1)
            # assert mean is not None and std is not None
            # W_2 = clip_state_dict[key].flatten(1)
            # std = torch.tensor(std).float()
            # std = std.repeat_interleave(clip_state_dict[key].shape[-1] * clip_state_dict[key].shape[-2])
            # W_1 = torch.diag(1 / std)
            # W_fused = W_2 @ W_1
            # mean = torch.tensor(mean).float().repeat_interleave(clip_state_dict[key].shape[-1] * clip_state_dict[key].shape[-2])
            # b_1 = mean / std
            # b_fused = W_2 @ (-b_1)
            # remapped_state_dict['visual.conv1.weight'] = W_fused
            # remapped_state_dict['visual.conv1.bias'] = b_fused
        elif key not in key_mapping:
            remapped_state_dict[key] = clip_state_dict[key]
        else:
            if key == 'positional_embedding':
                old_context_length, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                if context_length <= old_context_length:
                    remapped_state_dict[key_mapping[key]] = clip_state_dict[key][:context_length, :]
                else:
                    remapped_state_dict[key_mapping[key]] = torch.cat(
                        (clip_state_dict[key], torch.zeros((context_length - old_context_length, dim), dtype=old_dtype)), dim=0
                    )
            elif key == 'token_embedding.weight':
                old_vocab_size, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                assert vocab_size >= old_vocab_size
                remapped_state_dict[key_mapping[key]] = torch.cat(
                    (clip_state_dict[key], torch.zeros((vocab_size - old_vocab_size, dim), dtype=old_dtype)), dim=0
                )
            else:
                remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict

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
                scale=1., num_frames=8, drop_path=0.,
                shift: bool = False, 
                shift_type: str = 'psm',
                use_flash_attn: bool = False,
                ):
        super().__init__()

        self.use_flash_attn = use_flash_attn
        if shift:
            self.attn = FlashMHA(d_model, n_head, cross_attn=True, dropout=0., use_flash_attn=use_flash_attn)
        else:
            self.attn = FlashMHA(d_model, n_head, cross_attn=False, dropout=0., use_flash_attn=use_flash_attn)
        #     self.attn = nn.MultiheadAttention(d_model, n_head)
        
        self.ln_1 = LayerNorm(d_model)
        
        # if self.use_flash_attn:
        mlp_width = int(d_model * 4)
        self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=QuickGELU())
        # else:
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        
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
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        
        if self.shift:
            if self.shift_type == 'psm':
                # self.shift_op = PatchShift(self.n_head, False, 1)
                # self.shift_op_back = PatchShift(self.n_head, True, 1)
                self.shift_op = PatchShift_all(False)
                # self.shift_op_back = PatchShift_all(True)
            
            elif self.shift_type == 'tsm':
                self.shift_op = TemporalShift(8)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cross_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]
    
    # def attention(
    #     self, x: torch.Tensor , y: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
        
    #     # input: HW+2, BT, D
    #     q = (x @ self.attn.in_proj_weight[:self.d_model].T
    #          ) + self.attn.in_proj_bias[:self.d_model]

    #     k = (y @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
    #          ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
    #     v = (y @ self.attn.in_proj_weight[-self.d_model:].T
    #          ) + self.attn.in_proj_bias[-self.d_model:]
    #     Tx, Ty, N = q.size(0), k.size(0), q.size(1)
    #     q = q.view(Tx, N, self.attn.num_heads,
    #                self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
    #     k = k.view(Ty, N, self.attn.num_heads,
    #                self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
    #     v = v.view(Ty, N, self.attn.num_heads,
    #                self.attn.head_dim).permute(1, 2, 0, 3)
        
    #     aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)
        
    #     aff = aff.softmax(dim=-1)
        
        
    #     out = aff @ v  # N num_heads Tx D_head
    #     out = out.permute(2, 0, 1, 3).flatten(2)
    #     out = self.attn.out_proj(out) # N Tx D
        
    #     return out 
    
    def forward(self, x: torch.Tensor, ):
        ## x shape [ BT, HW+1, D]
        bt, n, d = x.shape
        ## temporal adaptation
        class_token=x[:, :1, :] # 1, BT, D
        
        xt = rearrange(class_token, '(b t) n d -> (b n) t d', t=self.num_frames)
        ln_xt=self.ln_1(xt)
        
        # if self.use_flash_attn:
        if self.shift:
            xt = self.T_Adapter(self.attn(x=ln_xt,x_kv=ln_xt))
        else:
            xt = self.T_Adapter(self.attn(ln_xt))
        
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=1)
        # x = x + self.drop_path(xt)
        x= torch.cat([x[:, :1, :], xt, x[:, 1:, :]], dim=1)# x shape [HW+2, BT, D]
        
        if self.shift:
            xln=self.ln_1(x)
            tmp_x = xln[ :, 2:, :].clone()
            
            NT, L,  C = tmp_x.shape
            T = self.num_frames
            N = NT // self.num_frames
            H = W = int(L**0.5)
            
            tmp_x = rearrange(tmp_x, '(b t) (h w) c -> b t h w c', b=N, t = T, h=H, w=W, c=C)
            tmp_x = self.shift_op(tmp_x)
            tmp_x = rearrange(tmp_x, 'b t h w c -> (b t) c h w')
            
            tmp_x = tmp_x.view(NT, C, -1).permute(0, 2, 1).contiguous() #  NT P C
            tmp_x = torch.cat([xln, tmp_x], dim=1)
            
            # if self.use_flash_attn:
            x = x + self.S_Adapter(self.attn(x=xln,x_kv=tmp_x)) 
            # else:
            #     x = x + self.S_Adapter(self.cross_attention(xln,tmp_x)) 
            
        else:
            # if self.use_flash_attn:
            x = x + self.drop_path(self.S_Adapter(self.attn(self.ln_1(x))))
            # else:
            #     x = x + self.drop_path(self.S_Adapter(self.attention(self.ln_1(x))))
        
        ## joint adaptation
        x= torch.cat([x[:, :1, :], x[:, 2:, :]], dim=1) # [HW+2, BT, D]
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x

class PatchShift(nn.Module):
    def __init__(self, n_div=8, inv=False, ratio=1):
        super(PatchShift, self).__init__()
        self.fold_div = n_div
        self.inv = inv
        self.ratio = ratio
        if inv:
            print(
                f'=> Using inverse PatchShift, head_num: {self.fold_div}, ratio {ratio}, tps'
            )
        else:
            print(f'=> Using PatchShift, head_num: {self.fold_div}, ratio {ratio}, tps')

    def forward(self, x):
        x = self.shift(x, fold_div=self.fold_div, inv=self.inv, ratio=self.ratio)
        return x #self.net(x)

    @staticmethod
    def shift(x, fold_div=3, inv=False, ratio=0.5):
        B, T ,num_heads, H, W, c = x.size()
        fold = int(num_heads * ratio)
        feat = x
        # feat = feat.view(B, T, num_heads, H, W, c)
        out = feat.clone()
        stride = 1
        multiplier = -1 if inv else 1
        ## Pattern C
        out[:, : , :fold, 0::3, 0::3,:] = torch.roll(feat[:, :,  :fold,0::3,0::3,:], shifts=-4*multiplier*stride, dims=1)
        out[:, : , :fold, 0::3, 1::3,:] = torch.roll(feat[:, :,  :fold,0::3,1::3,:], shifts=multiplier*stride, dims=1)
        out[:, : , :fold, 1::3, 0::3,:] = torch.roll(feat[:, :,  :fold,1::3,0::3,:], shifts=-multiplier*stride, dims=1)
        out[:, : , :fold, 0::3, 2::3,:] = torch.roll(feat[:, :,  :fold,0::3,2::3,:], shifts=2*multiplier*stride, dims=1)
        out[:, : , :fold, 2::3, 0::3,:] = torch.roll(feat[:, :,  :fold,2::3,0::3,:], shifts=-2*multiplier*stride, dims=1)
        out[:, : , :fold, 1::3, 2::3,:] = torch.roll(feat[:, :,  :fold,1::3,2::3,:], shifts=3*multiplier*stride, dims=1)
        out[:, : , :fold, 2::3, 1::3,:] = torch.roll(feat[:, :,  :fold,2::3,1::3,:], shifts=-3*multiplier*stride, dims=1)
        out[:, : , :fold, 2::3, 2::3,:] = torch.roll(feat[:, :,  :fold,2::3,2::3,:], shifts=4*multiplier*stride, dims=1) 

        # out = out.view(B, T ,num_heads, H, W, c)
        return out

class PatchShift_all(nn.Module):
    def __init__(self, inv=False):
        super(PatchShift_all, self).__init__()

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

class TemporalShift(nn.Module):
    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div
        print(f'=> Using channel shift, fold_div: {self.fold_div}')

    def forward(self, x):
        x = self.shift(x, fold_div=self.fold_div)
        return x

    @staticmethod
    def shift(x, fold_div=8):
        B, T ,num_heads, H, W, c = x.size()
        # B, T, num_heads, N, c = x.size()
        fold = c // fold_div
        feat = x
        feat = feat.view(B, T , num_heads, H*W, c)
        out = feat.clone()

        out[:, 1: , :, :, :fold] = feat[:, :-1, :, :, :fold]  # shift left
        out[:, :-1 , :, :, fold:2*fold] = feat[:, 1:, :, :, fold:2*fold]  # shift right 

        out = out.view(B, T ,num_heads, H, W, c)

        return out

class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                scale=1., drop_path=0.1, shift = False, shift_type: str = 'psm', use_flash_attn: bool = False):
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
                    shift= shift,
                    shift_type=shift_type,
                    use_flash_attn=use_flash_attn
                )
                for i in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

@MODELS.register_module()
class ViT_CLIP_FLASH(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, 
                adapter_scale=0.5, 
                pretrained=None, 
                shift = False,
                shift_type='psm',
                use_flash_attn: bool = False ):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.layers = layers
        
        self.shift=shift
        self.width=width
        
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        
        
        self.transformer = Transformer(num_frames, width, layers, heads, scale=adapter_scale, drop_path=drop_path_rate,
                                        shift=shift ,shift_type=shift_type,
                                        use_flash_attn=use_flash_attn)

        self.ln_post = LayerNorm(width)
        
        self.use_flash_attn = use_flash_attn

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
            
            # remapped_state_dict = remap_keys_from_open_clip_to_vit(
            #     clip_model.state_dict(),
            #     use_flash_attn=self.use_flash_attn,
            # )
            
            # for key in remapped_state_dict.keys():
            #     print(key)
            
            # pretrain_dict = remapped_state_dict['visual']
            # del remapped_state_dict
            
            # 'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
            #     'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
            #     'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
            #     'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
            
            if not self.shift:
                swaps = [('attn.in_proj_weight', 'attn.Wqkv.weight'), ('attn.in_proj_bias', 'attn.Wqkv.bias'),
                    ('attn.out_proj.weight','attn.out_proj.weight'),('attn.out_proj.bias','attn.out_proj.bias'),
                    ('mlp.c_fc.weight','mlp.fc1.weight'),('mlp.c_fc.bias','mlp.fc1.bias'),
                    ('mlp.c_proj.weight','mlp.fc2.weight'),('mlp.c_proj.bias','mlp.fc2.bias')]
            else:
                swaps = [('attn.in_proj_weight', 'attn.Wq.weight','attn.Wkv.weight'), ('attn.in_proj_bias', 'attn.Wq.bias','attn.Wkv.bias'),
                    ('attn.out_proj.weight','attn.out_proj.weight'),('attn.out_proj.bias','attn.out_proj.bias'),
                    ('mlp.c_fc.weight','mlp.fc1.weight'),('mlp.c_fc.bias','mlp.fc1.bias'),
                    ('mlp.c_proj.weight','mlp.fc2.weight'),('mlp.c_proj.bias','mlp.fc2.bias')]
            
            out_dict={}
            for k, v in pretrain_dict.items():
                flag=True
                for sp in swaps:
                    if sp[0] in k:
                        if len(sp)==2:
                            k = k.replace(sp[0], sp[1])
                            out_dict[k] = v
                        else:
                            k2=k
                            k = k.replace(sp[0], sp[1])
                            out_dict[k] = v[:self.width]
                            k2 = k2.replace(sp[0], sp[2])
                            out_dict[k2] = v[self.width:]
                        flag=False
                if flag:
                    out_dict[k]=v
            
            msg = self.load_state_dict(out_dict, strict=False)
            
            
            # msg = self.load_state_dict(pretrain_dict, strict=False)
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

        logger.info('freeze the backbone of ViT_CLIP_TPS')
        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name :
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info(
            f'Number of total parameters: {(num_total_param/1.e6):6.2f}, tunable parameters: {(num_param/1.e6):6.2f}'
        )
        

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

        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
    

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(ViT_CLIP_FLASH, self).train(mode)
        

if __name__ == '__main__':

    from mmengine.analysis import get_model_complexity_info

    backbone=dict(
        type='ViT_CLIP_FLASH',
        pretrained='openaiclip',
        input_resolution=224,
        patch_size=16,
        num_frames=32,
        width=768,
        layers=12,
        heads=12,
        drop_path_rate=0.1),

    input_shape = (3,32,224,224)
    model = ViT_CLIP_FLASH(pretrained='openaiclip',
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
