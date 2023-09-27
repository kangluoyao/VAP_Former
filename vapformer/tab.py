import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import reduce
from operator import mul
from einops import rearrange, repeat
import math

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # print(q.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # print(q.shape)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

# embeddings for tabular

class Embeddings(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.apoe = num_tokens - 3
        self.embeds1 = nn.Parameter(torch.randn(self.apoe, dim))
        self.embeds2 = nn.Linear(1,dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):

        x1 = torch.mul(x[:,0:self.apoe,:], self.embeds1)
        x2 = self.embeds2(x[:,self.apoe:,:])
        
        x = torch.cat([x1,x2],dim=1)
        cls_token = self.avgpool(x)
        
        x = torch.cat([cls_token, x], dim=1)
        
        return x


# embeddings2 
class Embeddings2(nn.Module):
    def __init__(self, vis_patch_num, tab_patch_num, dim):
        super().__init__()
        self.num_tokens = vis_patch_num + tab_patch_num
        self.embeds1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        x = self.norm1(x)
        x = self.embeds1(x)
        x = self.norm2(x)
        return x

# transformer_encoder
class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.attn = Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)))
        self.ff = Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout)))

    def forward(self, x): 

        x = self.attn(x)
        x = self.ff(x)
        
        return x

class Transformer_encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, vis):
        super().__init__()
        self.vis = vis
        self.layers = nn.ModuleList([Block(dim, heads, dim_head, attn_dropout, ff_dropout) for _ in range(depth)])

    def forward(self, x): 

        for layer in self.layers:
            x = layer(x)
        
        return x

# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, vis):
        super().__init__()
        self.embeddings = Embeddings(num_tokens, dim)

        self.encoder = Transformer_encoder(dim, depth, heads, dim_head, attn_dropout, ff_dropout, vis)

    def forward(self, x):

        x = self.embeddings(x)
        
        x = self.encoder(x)
        
        return x

# ConcatTransformer
class ConcatTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, vis, vis_patch_num, tab_patch_num, emb_dropout, num_classes, pool = 'cls'):
        super().__init__()
        self.embeddings = Embeddings2(vis_patch_num, tab_patch_num, dim)

        self.encoder = Transformer_encoder(dim, depth, heads, dim_head, attn_dropout, ff_dropout, vis)

        self.pos_embedding = nn.Parameter(torch.randn(1, vis_patch_num + tab_patch_num + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.sig = nn.Sigmoid()
        self.pool = pool
    
    def train(self, mode=True):
        if mode:
            # training:
            self.embeddings.eval()
            self.encoder.eval()
            self.dropout.eval()
            self.to_latent.train()
            self.mlp_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode) 


    def forward(self, x):
        # print(x.shape)
        x = self.embeddings(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.encoder(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.sig(x)
        return x

class PromptTransformer(Transformer):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, prompt_dropout, prompt_dim, vis, prompt_num_tokens=50):
        super(PromptTransformer, self).__init__(num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, vis)
        self.depth = depth
        self.prompt_dim = prompt_dim
        self.prompt_num_tokens = prompt_num_tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        self.prompt_proj = nn.Linear(prompt_dim, dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')

        #initiate prompt
        val = math.sqrt(6. / float(3 * reduce(mul, (1, 1), 1) + prompt_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        # deep prompt
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(depth-1, prompt_num_tokens, prompt_dim))
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)   

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.depth

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.encoder.layers[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.prompt_num_tokens):, :]
                    ), dim=1)


                hidden_states = self.encoder.layers[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = hidden_states
        return encoded

    def forward(self, x):

        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        #deep prompt version:
        encoded = self.forward_deep_prompt(embedding_output)
        # shallow prompt version:
        # encoded = self.encoder(embedding_output)

        return encoded
        




# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# # main class

# class TabTransformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         num_features,
#         dim,
#         depth,
#         heads,
#         dim_head = 16,
#         attn_dropout = 0.,
#         ff_dropout = 0.,
#         vis = False
#     ):
#         super().__init__()
#         # categories related calculations
#         self.num_features = num_features

#         # transformer
#         self.transformer = Transformer(
#             num_tokens = self.num_features,
#             dim = dim,
#             depth = depth,
#             heads = heads,
#             dim_head = dim_head,
#             attn_dropout = attn_dropout,
#             ff_dropout = ff_dropout,
#             vis = vis
#         )

#     def forward(self, x):
#         x = self.transformer(x)
        
#         return x


# class PromptTabTransformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         num_features,
#         dim,
#         depth,
#         heads,
#         dim_head = 16,
#         attn_dropout = 0.1,
#         ff_dropout = 0.1,
#         prompt_dropout = 0.1,
#         prompt_dim = 512,
#         prompt_num_tokens = 50,
#         vis = False
#     ):
#         super().__init__()
#         # categories related calculations
#         self.num_features = num_features

#         # transformer
#         self.transformer = PromptTransformer(
#             num_tokens = self.num_features,
#             dim = dim,
#             depth = depth,
#             heads = heads,
#             dim_head = dim_head,
#             attn_dropout = attn_dropout,
#             ff_dropout = ff_dropout,
#             prompt_dropout = prompt_dropout,
#             prompt_dim = prompt_dim,
#             vis = vis,
#             prompt_num_tokens = prompt_num_tokens,
#         )

#     def forward(self, x):
#         x = self.transformer(x)
        
#         return x