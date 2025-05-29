# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from re import S
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import random
from timm.models.vision_transformer import Mlp, PatchEmbed
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    pass


# for i in sys.path:
#     print(i)

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # Flash Attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).contiguous()
            attn_output = flash_attn_qkvpacked_func(qkv,causal=False,dropout_p=0.1)
            # attn_output = attn_output.permute(0, 2, 1, 3) # (b,nh, l, c) -> (b, l, nh, c
            x = attn_output.reshape(B, N, C)

        elif self.attention_mode == 'math':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


# 将时间步标量转换为向量表示
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb

# 将类别标签标量转换为向量表示
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        # 无分类器指导
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core IRASim Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A IRASim tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of IRASim.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class IRASim(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32, # input size是的是啥，可能是长宽或者高，等下将会传入patch embed
        patch_size=2, # patch_size 
        in_channels=4, # in channels 指的是啥， 可能是指RGBA的通道？
        hidden_size=1152, # hidden_size 指的是啥
        depth=28, # depth 指的是啥
        num_heads=16, # 这个应该是transformer的head
        mlp_ratio=4.0, # 这是啥啊, 其实这个决定了Transformer Block 的MLP的HiddenDim的大小
        num_frames=16, # 输入的视频帧数
        learn_sigma=True,
        extras=1, # 这个extra到底是啥意思？ -> 3: frame level condition, 5: video level condition
        attention_mode='math',
        args = None
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames
        self.args = args

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 3: # frame level
            if args.dataset == 'languagetable':
                self.state_dim = 2
                self.embed_arm_state = torch.nn.Linear(self.state_dim, 4*hidden_size)
                self.embed_state = torch.nn.Linear(4* hidden_size, hidden_size)
                self.mask_emb_fn = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
            elif args.dataset == 'rt1': # or args.dataset == 'bridge':
                self.state_dim = 7 
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.embed_state = Mlp(in_features=self.state_dim, hidden_features = hidden_size*4, out_features=hidden_size, act_layer=approx_gelu, drop=0)
                self.mask_emb_fn = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
            elif args.dataset == 'bridge':
                self.state_dim = 7 # 这个应该值得是 bridge 的state
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.embed_state = Mlp(in_features=self.state_dim, hidden_features = hidden_size*4, out_features=hidden_size, act_layer=approx_gelu, drop=0)
                self.mask_emb_fn = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
        elif self.extras == 5:
            if args.dataset == 'languagetable':
                self.state_dim = 2
                self.embed_arm_state = torch.nn.Linear(self.state_dim, 4*hidden_size)
                self.embed_state = torch.nn.Linear(4 * hidden_size, hidden_size)
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(in_features=(num_frames-1)*hidden_size , out_features=hidden_size, act_layer=approx_gelu, drop=0)
                self.mask_emb_fn = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
            else:
                self.state_dim = 7
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.embed_state = Mlp(in_features=(num_frames-1)*self.state_dim, hidden_features = hidden_size*4, out_features=hidden_size, act_layer=approx_gelu, drop=0)
                self.mask_emb_fn = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        # 这两个都不训练的
        """
        对于pos_embedding
        shape 为 (1, num_patches, hidden_size)
        一帧图片上的每个 patch 会得到一个 hidden_size 维的向量
        
        对于temp_embdding
        shape 为 (1, num_frames, hidden_size)
        每一帧会得到一个hidden_size维的向量
        """
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        pos_embed = get_2d_sincos_pos_embed_non_square(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in IRASim blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        N: Batch size T: patch 的数量 patchsize**2*c: 单个patchsize的像素数
        imgs: (N, H, W, C)
        N: Batch size H: 高， W: 宽, C: 通道数
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        h, w = self.x_embedder.grid_size # grid size是指patch 的 grid size
        assert h * w == x.shape[1] # 这句话是啥意思？

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                t, 
                actions=None,
                text_embedding=None, 
                mask_frame_num = None,
                use_fp16=False):
        """
        Forward pass of IRASim.
        x: (N, F, C, H, W) tensor of video inputs
        N -> Batch size
        F -> Frames
        C -> Channels
        H -> Height
        W -> Weight
        t: (N,) tensor of diffusion timesteps
        N -> Batch Size
        
        如果没有猜错的话，这个Action的形状应该是
        (Batchsize, Frames, ACtionDim)
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
        print("======")
        
        print(f"raw x > {x.shape=}")
        print(f"raw t > {t.shape=}")
        print(f"raw Action > {actions.shape=}") 
        
        batches, frames, channels, high, weight = x.shape 
        print(f"frames > {frames}")
        print(f"batch > {batches}")
        print(f"channels > {channels}")
        print(f"high > {high}")
        print(f"weight > {weight}")
        print(f"num_patches > {self.x_embedder.num_patches}")
        print(f"hidden_size > {self.hidden_size}") 
        do_cfg = False

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        
        print("======")
        print(f"rearrange > {x.shape=}")
        """
        假设我们有一个形状为 (2, 3, 64, 32, 32) 的张量，表示 2 个样本，每个样本有 3 帧，且每帧是 64 通道，
        32x32 像素。使用 rearrange 后，形状变为 (6, 64, 32, 32)，这意味着我们将原来
        每个样本的 3 帧合并为 1 个维度，并且每个样本和每帧被看作独立的处理单元。这样就可以并行处理帧了，（比如说Patchify处理）
        这个操作之后， 已经可以以frame作为索引了
        """
        
        x = self.x_embedder(x) 
        print(f"pos_embed > {self.pos_embed.shape}")
        print(f"temp_embed > {self.temp_embed.shape}")
        print(f"after embedder > {x.shape=}")
        x = x + self.pos_embed
        t = self.t_embedder(t, use_fp16=use_fp16) 
        print(f"after embedder > {t.shape=}")             
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=frames) 
        print(f"timestep_spatial > {timestep_spatial.shape=}")
        # 也就是对时间步embedding 沿 frams维度复制 frames 次。
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        print(f"timestep_temp > {timestep_temp.shape=}")
        # 也就是对时间步embedding 沿 frams维度复制 frames 次。
        """
            对于 timestep_spatial
            N = 2（两个视频样本），
            frames = 4（每个样本4帧），
            D = 128（时间嵌入维度）
            那 t 的 shape 是 (2, 128)，重复后就变成了 (2 × 4, 128) = (8, 128)。
            所以 timestep_spatial 是给 每一帧的空间编码 搭配的时间嵌入。
            对于 timestep_temp
            这
            
        """
        
        if self.extras == 3:
            if self.args.dataset == 'languagetable':
                arm_state = actions
                state_embeddings = self.embed_arm_state(arm_state) 
                state_embeddings = self.embed_state(state_embeddings)
            elif self.args.dataset == 'rt1' :# or self.args.dataset == 'bridge':
                state_embeddings = self.embed_state(actions)
            elif self.args.dataset == 'bridge':
                # print("ok")
                state_embeddings = self.embed_state(actions)
                print(f"embedded action > {state_embeddings.shape}")

            mask_emb = self.mask_emb_fn(torch.tensor(0, device=state_embeddings.device))
            batch_front_mask_emb = repeat(mask_emb, 'd -> b 1 d', b = state_embeddings.size()[0])
            print(f"batch_front_mask_emb > {batch_front_mask_emb.shape}")
            state_embeddings = torch.cat([
                batch_front_mask_emb,
                state_embeddings
                ], dim=1) # 为啥啊，为啥要往前面Attach一个特殊的token，可以叫token吗？
            print(f"state_embeddings > {state_embeddings.shape}")
            """
            GPT的解释:
            这种方式通常用于一些序列或结构化数据的模型中，
            在序列的开头（或其他位置）引入特殊的标记（如 mask，cls 等）以便模型能够正确地处理和区分这些位置。
            """

            if self.training:
                mask_emb_expanded = batch_front_mask_emb.expand_as(state_embeddings) 
                mask = torch.rand(batches,device=state_embeddings.device) < 0.1
                state_embeddings[mask] = mask_emb_expanded[mask]    
            state_embeddings = state_embeddings.reshape(-1,state_embeddings.size()[-1])
            # 重新沾回去 （Batchsize， dim）的形式
            print(f"state_embeddings mask_emb added > {state_embeddings.shape}")
            # print(f"temporal_state_embeddings > {state_embeddings}")
        elif self.extras == 5:
            if self.args.dataset == 'languagetable':
                arm_state = actions
                state_embeddings = self.embed_arm_state(arm_state) 
                state_embeddings = self.embed_state(state_embeddings) # b,len,h
                state_embeddings = self.mlp(state_embeddings.reshape(batches,-1))
            else:
                actions = actions.reshape(actions.size()[0],-1)
                state_embeddings = self.embed_state(actions)
            if self.training:
                mask = torch.rand(state_embeddings.size(0),device=state_embeddings.device) < 0.1
                mask_emb = self.mask_emb_fn(torch.tensor(0,device=state_embeddings.device))
                mask_emb_expanded = mask_emb.unsqueeze(0).expand_as(state_embeddings) 
                state_embeddings[mask] = mask_emb_expanded[mask]    
            # print(f"state_embeddings > {state_embeddings}")
            
            spatial_state_embeddings = repeat(state_embeddings, 'n d -> (n c) d', c=frames) 
            temporal_state_embeddings = repeat(state_embeddings, 'n d -> (n c) d', c=self.pos_embed.shape[1])

            
        
        # 到目前为止各种embeddings好像都搞完了
        
        for i in range(0, len(self.blocks), 2):
            ## 空间块和时间块是交替存在
            ## 交替复制时间空间embeddings
            spatial_block, temp_block = self.blocks[i:i+2]
            # 取Transformer的第
            if self.extras == 3:
                c = timestep_spatial + state_embeddings
                print(f"c = timestep_spatial + state_embeddings > {c.shape}")
            elif self.extras == 5:
                c = timestep_spatial + spatial_state_embeddings
            else:
                c = timestep_spatial
            print(f"Spacial > [B*F, P, D] > {x.shape=}")
            x  = spatial_block(x, c)

            
            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed[:,0:frames]
                print(f"x + temp_embed > {x.shape}")

            if self.extras == 3:
                c = timestep_temp
                print(f"c = timestep_temp > {c.shape}")
            elif self.extras == 5:
                c = timestep_temp + temporal_state_embeddings
            else:
                c = timestep_temp
            print(f"Temporal > [B*P, F, D] > {x.shape=}")
            x = temp_block(x, c)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)
            
        if self.extras == 5:
            c = timestep_spatial + spatial_state_embeddings
        elif self.args.final_frame_ada:
            # this is a legacy bug, the given checkpoint for IRASim-Frame-Ada are trained without last ada, but it should be
            c = timestep_spatial + state_embeddings
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)               
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed_non_square(embed_dim, grid, cls_token=False, extra_tokens=0):
    """
    grid_h_size: int, height of the grid
    grid_w_size: int, width of the grid
    return:
    pos_embed: [grid_h_size*grid_w_size, embed_dim] or [1+extra_tokens+grid_h_size*grid_w_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h_size, grid_w_size = grid
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed
    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   IRASim Configs                                  #
#################################################################################

def IRASim_XL_2(**kwargs):
    return IRASim(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def IRASim_XL_4(**kwargs):
    return IRASim(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def IRASim_XL_8(**kwargs):
    return IRASim(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def IRASim_L_2(**kwargs):
    return IRASim(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def IRASim_L_4(**kwargs):
    return IRASim(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def IRASim_L_8(**kwargs):
    return IRASim(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def IRASim_B_2(**kwargs):
    return IRASim(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def IRASim_B_4(**kwargs):
    return IRASim(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def IRASim_B_8(**kwargs):
    return IRASim(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def IRASim_S_2(**kwargs):
    return IRASim(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def IRASim_S_4(**kwargs):
    return IRASim(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def IRASim_S_8(**kwargs):
    return IRASim(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


IRASim_models = {
    'IRASim-XL/2': IRASim_XL_2,  'IRASim-XL/4': IRASim_XL_4,  'IRASim-XL/8': IRASim_XL_8,
    'IRASim-L/2':  IRASim_L_2,   'IRASim-L/4':  IRASim_L_4,   'IRASim-L/8':  IRASim_L_8,
    'IRASim-B/2':  IRASim_B_2,   'IRASim-B/4':  IRASim_B_4,   'IRASim-B/8':  IRASim_B_8,
    'IRASim-S/2':  IRASim_S_2,   'IRASim-S/4':  IRASim_S_4,   'IRASim-S/8':  IRASim_S_8,
}


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
if __name__ == '__main__':

    import torch
    import yaml
    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = '/home/junzhicai/workspace/IRASim/configs/train/bridge/frame_ada.yaml'  # Your config file path
    config_dict = load_config(config_path)
    config = Config(config_dict)
    img = torch.randn(3, 16, 4, 32, 32, dtype=torch.float32, device=device)
    t = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    y = torch.randn((3, 15, 7), dtype=torch.float32, device=device)


    network = IRASim_S_2(args=config, extras=3).to(device)
    # from thop import profile 
    # flops, params = profile(network, inputs=(img, t))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    out = network(img, t, y)
    print(out.shape)