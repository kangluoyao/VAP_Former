
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class InterMultiHeadAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim,  num_features,num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, clinical, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww*Wz, Wh*Ww*Wz, ) or None
        """
        # print(x.shape)
        # print(clinical.shape)
        B_, N, C = x.shape
        kv = self.kv(clinical).reshape(B_, self.num_features, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        # q = self.q(x).permute(0,2,1).reshape(B_, self.num_heads, N, C // self.num_heads)
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        
        x = x.reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class IntraMultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = dim
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(dim, dim * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
class ResidualAdd2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class InterTransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_features,attn_dropout = 0.1,
        ff_dropout = 0.1):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.LN = nn.LayerNorm(dim)
        self.MSA = InterMultiHeadAttention(dim = dim, num_features=num_features, attn_drop = attn_dropout)
        self.FFN = FeedForwardBlock(emb_size = dim, drop_p = ff_dropout)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(dim*2,dim*2)


    def forward(self, x, clinical):
        full_clinical = clinical.permute(1, 0, 2)[1:self.num_features+1].permute(1, 0, 2)
        global_clinical = clinical.permute(1, 0, 2)[0].unsqueeze(0).permute(1, 0, 2)
        x = x.permute(1, 0, 2)[1:].permute(1, 0, 2)
        global_img = x.permute(1, 0, 2)[0].unsqueeze(0).permute(1, 0, 2)
        global_va = torch.cat([global_clinical, global_img], dim=2)
        
        betaalpha = self.fc1(global_va).reshape(-1,2,self.dim).permute(1, 0, 2)
        alpha = betaalpha[0].unsqueeze(1)
        beta = betaalpha[1].unsqueeze(1)
        res = x
        
        x = self.LN(x)
        x = self.MSA(x,full_clinical)
        x = res * x
        
        res = x
        x = self.LN(x)
        x = self.FFN(x)
        x += res
        x = x*alpha + beta
        x = torch.cat([global_img, x], dim=1)
        return x
    
class IntraTransformerEncoderBlock(nn.Module):
    def __init__(self, dim,attn_dropout = 0.1,
        ff_dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.LN = nn.LayerNorm(dim)
        self.MSA = IntraMultiHeadAttention(dim = dim, dropout = attn_dropout)
        self.FFN = FeedForwardBlock(emb_size = dim, drop_p = ff_dropout)


    def forward(self, x):
        res = x
        x = self.LN(x)
        x = self.MSA(x)
        x = res + x
        res = x
        x = self.LN(x)
        x = self.FFN(x)
        x += res
        
        return x

class LGDFblock(nn.Module):
    def __init__(self, dim, num_features,
        num_patches = 80,
        heads = 8,
        depth = 4,
        attn_dropout = 0.1,
        ff_dropout = 0.1):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.dim = dim
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.VisionInterBlock = InterTransformerEncoderBlock(dim = dim, num_features=num_features,attn_dropout = attn_dropout,ff_dropout = ff_dropout)
        self.TabularInterBlock = InterTransformerEncoderBlock(dim = dim, num_features=num_patches,attn_dropout = attn_dropout,ff_dropout = ff_dropout)
        self.IntraBlock = IntraTransformerEncoderBlock(dim = dim, attn_dropout = attn_dropout,ff_dropout = ff_dropout)
        self.IntraBlock2 = IntraTransformerEncoderBlock(dim = dim, attn_dropout = attn_dropout,ff_dropout = ff_dropout)



    def forward(self, x, clinical):
        
        res = x
        x = self.VisionInterBlock(x, clinical)
        x = self.IntraBlock(x)
        clinical = self.TabularInterBlock(clinical, res)
        clinical = self.IntraBlock2(clinical)
        
        
        return x,clinical




class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_features,
        num_patches = 80,
        heads = 8,
        depth = 4,
        attn_dropout = 0.1,
        ff_dropout = 0.1):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.dim = dim
        self.LGC_layers = nn.ModuleList(
            [LGDFblock(dim=dim, num_features=num_features, num_patches=num_patches, heads=heads, attn_dropout=attn_dropout,ff_dropout=ff_dropout) for _ in range(self.depth)]
        )



    def forward(self, x, clinical):
        # print("x",x.shape)    # (b, 80, 256)
        # print("clinical",clinical.shape)   # (b, 15, 256)
        
        for layer_module in self.LGC_layers:
            x, clinical = layer_module(x, clinical)


        return x,clinical
        
        

