from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from .layers import LayerNorm
from .transformerblock import TransformerBlock, PromptTransformerBlock
from .dynunet_block import get_conv_layer, UnetResBlock
from .IntraAtt import TransformerEncoder
from .tab import Transformer, PromptTransformer, ConcatTransformer
import torch
import torch.nn.functional as F
import math
from torch.nn import Dropout
from functools import reduce
from operator import mul



einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(3, 3, 3), stride=(3, 3, 3),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        # hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # if i == 3:  # Reshape the output of the last stage
            #     x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            # hidden_states.append(x)
        return x
        # return x, hidden_states

    def forward(self, x):
        img_features = self.forward_features(x)
        return img_features

class PromptUnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(3, 3, 3), stride=(3, 3, 3),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if i < 3:
                    stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
                else:
                    stage_blocks.append(PromptTransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def train(self, mode=True):
        if mode:
            self.downsample_layers.eval()
            self.stages[0].eval()
            self.stages[1].eval()
            self.stages[2].eval()
            self.stages[3].train()
            print("PromptUnetrPPEncoder train mode")
        else:
            # eval:
            for module in self.children():
                module.train(mode)



    def forward_features(self, x):
        # hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        # hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # if i == 3:  # Reshape the output of the last stage
            #     x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            # hidden_states.append(x)
        return x
        # return x, hidden_states

    def forward(self, x):
        img_features = self.forward_features(x)
        return img_features


class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class thenet(nn.Module):
    def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=8, spatial_dims=3, in_channels=1,
                 dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
        super().__init__()
        self.dims = dims
        self.img_encoder = PromptUnetrPPEncoder(
                        input_size=input_size,
                        dims=dims, 
                        depths=depths, 
                        num_heads=num_heads,
                        in_channels=in_channels,
                        proj_size = proj_size,
                        spatial_dims=spatial_dims,
                        dropout=dropout,
                        transformer_dropout_rate=transformer_dropout_rate
                        )

        self.sig  = nn.Sigmoid()
        self.classifier1 = nn.Linear(dims[3], 1)

        self.tabformer = PromptTransformer(
            num_tokens = 9,                  # number of features, paper set at 512   调整特征数要调整这里
            dim = 256,                           # dimension, paper set at 512
            dim_head= 16,
            depth = 3,                          # depth, paper recommended 3
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.1,                 # post-attention dropout
            ff_dropout = 0.1,                   # feed forward dropout
            prompt_dropout = 0.1,
            prompt_num_tokens = 100,
            prompt_dim = 512, 
            vis = False
        )

        self.ConcatTransformer = ConcatTransformer(
            dim = 256, 
            depth = 3, 
            heads = 8, 
            dim_head = 16, 
            attn_dropout = 0.1, 
            ff_dropout = 0.1, 
            vis = False, 
            vis_patch_num = 80, 
            tab_patch_num = 9, 
            emb_dropout = 0.1, 
            num_classes = 1
        )


        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def train(self, mode=True):
        if mode:
            # training:
            self.tabformer.train()
            self.img_encoder.train()
            self.ConcatTransformer.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode) 
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, clinical):
        img_features = self.img_encoder(x)
        B,C,H,W,D = img_features.shape 
        img_features = img_features.view(B,C,-1).permute(0,2,1)
        clinical = self.tabformer(clinical.unsqueeze(2))
        clinical = clinical[:, -9:, :]
        fuse_features = torch.cat((img_features, clinical), dim=1)
        out = self.ConcatTransformer(fuse_features)
        
        return out


# class Promptthenet(thenet):
#     def __init__(self, input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],dims=[32, 64, 128, 256],
#                  proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=8, spatial_dims=3, in_channels=1,
#                  dropout=0.2, transformer_dropout_rate=0.2 ,**kwargs):
#         super().__init__()
#         img_size = _pair(img_size)
#         patch_size = _pair(config.patches["size"])

#         num_tokens = self.prompt_config.NUM_TOKENS
#         self.num_tokens = num_tokens  # number of prompted tokens

#         self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

#         # if project the prompt embeddings
#         if self.prompt_config.PROJECT > -1:
#             # only for prepend / add
#             prompt_dim = self.prompt_config.PROJECT
#             self.prompt_proj = nn.Linear(
#                 prompt_dim, config.hidden_size)
#             nn.init.kaiming_normal_(
#                 self.prompt_proj.weight, a=0, mode='fan_out')
#         else:
#             prompt_dim = config.hidden_size
#             self.prompt_proj = nn.Identity()

#         # initiate prompt:
#         if self.prompt_config.INITIATION == "random":
#             val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

#             self.prompt_embeddings = nn.Parameter(torch.zeros(
#                 1, num_tokens, prompt_dim))
#             # xavier_uniform initialization
#             nn.init.uniform_(self.prompt_embeddings.data, -val, val)

#             if self.prompt_config.DEEP:  # noqa

#                 total_d_layer = config.transformer["num_layers"]-1
#                 self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
#                     total_d_layer, num_tokens, prompt_dim))
#                 # xavier_uniform initialization
#                 nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)


#     def incorporate_prompt(self, x):
#         # combine prompt embeddings with image-patch embeddings
#         B = x.shape[0]
#         # after CLS token, all before image patches
#         x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
#         x = torch.cat((
#                 x[:, :1, :],
#                 self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
#                 x[:, 1:, :]
#             ), dim=1)
#         # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

#         return x
        
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (LayerNorm, nn.LayerNorm)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)


#     def forward(self, x, clinical):
#         img_features = self.img_encoder(x)

#         clinical = self.tabformer(clinical.unsqueeze(2))
#         clinical = clinical.view(clinical.size(0), -1)

#         img_features = self.DAFTmodel(img_features,clinical)

#         out = F.relu(img_features, inplace=True)
#         out = F.adaptive_avg_pool3d(out, 1)
#         out = torch.flatten(out, 1)
#         out = self.classifier1(out)
#         out = self.sig(out)
        
#         return out