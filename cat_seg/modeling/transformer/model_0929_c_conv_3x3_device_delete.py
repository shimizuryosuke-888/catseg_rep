# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Licensed under The MIT License [see LICENSE for details]
# Written by Seokju Cho and Heeseong Shin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

# 6.1 ppg_import
# 修正後
from . import ppg_0816_reshape as ppg

from ...utils.debug import dbg

# 7.1 build_sam_blockのimport
# 修正後
from . import sam_block

# Modified Swin Transformer blocks for guidance implementetion
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        # 0917_1.2_self.vの線形変換をCとFv全体に対して行う．
        # 修正前
        # self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        # 修正後
        self.v = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        # ---------------------------------
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # 0917_1.2_self.vの線形変換をCとFv全体に対して行う
        # 修正前
        # v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # 修正後
        v = self.v(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # ------------------------------

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 0917_1.1_正規化後のxをshortcut
        # 修正前
        # shortcut = x
        # x = self.norm1(x)
        # 修正後
        x = self.norm1(x)
        shortcut = x
        # -----------------------------
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlockWrapper(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5, pad_len=0):
        super().__init__()
        self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=0)
        # 0911_Wrapper_review_escblock_1, SwinWrapperのSwinBlock2削除
        # 修正後
        # self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
                # 7.7 samblock後のappearance_guidanceが，(4, 256, 24, 24)から(4, 256, 171, 24, 24)
                appearance_guidance: B C T H W
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        # 7.7 samblock後のappearance_guidanceが，(4, 256, 24, 24)から(4, 256, 171, 24, 24)
        if appearance_guidance is not None:
            # appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
            appearance_guidance = self.guidance_norm(rearrange(appearance_guidance, "B C T H W -> (B T) (H W) C", B=B, H=H))
        # --------------------------------
        x = self.block_1(x, appearance_guidance)
        # 0911_Wrapper_review_escblock_1, SwinWrapperのSwinBlock2削除
        # 修正後
        # x = self.block_2(x, appearance_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        # 3.1_qkvの次元調整
        # 修正前
        # self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        # self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        # self.v = nn.Linear(hidden_dim, hidden_dim)
        # 修正後
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(guidance_dim, hidden_dim)
        self.v = nn.Linear(guidance_dim, hidden_dim)
        # ----------------------------

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        # 3.2_torchcat変更
        # 修正前
        # q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        # k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        # v = self.v(x)
        # 修正後
        q = self.q(x)
        k = self.k(guidance)
        v = self.v(guidance)
        # ---------------------------

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4), pad_len=256) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()
        
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H_pool, W=W_pool)
                x_pool = torch.cat([x_pool, padding_tokens], dim=2)

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        
        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        # 0917_2.4_MLP後の残差を削除
        # 修正前
        # x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP
        # 修正後
        x_pool = self.MLP(self.norm2(x_pool)) # MLP
        # -----------------------

        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        # 0917_2.2_pool後の特徴マップbilinear
        # 修正前
        # x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        # 修正後
        # --------------------------
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        if self.padding_tokens is not None:
            if orig_len < self.pad_len:
                x_pool = x_pool[:, :, :orig_len]

        # 0917_2.5_最後の残差を削除
        # 修正前
        # x = x + x_pool # Residual
        # 修正後
        # --------------
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AggregatorLayer(nn.Module):
    def __init__(self, 
                hidden_dim=64, 
                text_guidance_dim=512, 
                appearance_guidance=512, 
                nheads=4, 
                input_resolution=(24, 24), 
                pooling_size=(5, 5), 
                window_size=(10, 10), 
                attention_type='linear', 
                pad_len=256, 
                prompt_channel=1, 
                last_block=False) -> None:
        super().__init__()
        # 1.6 畳み込みinput_channels prompt_channels，最終ESCBlockフラグ last_block=True or False 受け取り変数追加
        # 修正前
        # self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256
        # 修正後
        # self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256, prompt_channels=1, last_block=False 
        # ----------------------

        self.swin_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)


        
        # 1.6.1 最終ESCBlockフラグself登録
        self.last_block = last_block
        # ----------------------

        # 6.5 ppgに必要な変数をinitする．
        # 修正後
        self.k_pts = 5
        self.ppg_thr = 0.0018
        self.clip_resolution = (336, 336)
        self.ppg_kmeans_iters = 7
        # ----------------------

        # 7.2 AggregatorLayer initでのbuild_sam_block
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.sam_block = sam_block.build_sam_block().to(self.device)
        self.sam_block = sam_block.build_sam_block()
        self.sam_block.prompt_encoder.image_embedding_size = (24, 24)
        self.sam_block.input_image_size = (336, 336)
        self.sam_chunk = 128
        self.H_emb, self.W_emb = input_resolution

        # 0911_Wrapper_review_escblock_1, 1.5 corrの畳み込みconv2d追加
        # 修正前
        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        # 修正後
        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=1).to(self.device)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=1)
        # ----------------------

        # 4.1.1_samblock_1x1conv_init
        # 修正後
        # self.sam_prompts_conv = nn.Conv2d(self.k_pts*256, 256, kernel_size=1).to(self.device)
        self.sam_prompts_conv = nn.Conv2d(self.k_pts*256, 256, kernel_size=1)
        # ----------
        self.c_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, padding=1, bias=True, padding_mode="replicate")

    
    # 1.7 corr畳み込み処理のwrapper関数の追加
    # 修正後
    
    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
        
    # ----------------------
    @torch.no_grad()
    def _run_ppg(self, corr_grid: torch.Tensor):
        """
        ====== PPG (領域マスク＆点) ======
        元コードの分岐をそのまま踏襲（batched or vectorized）
        """

        coords, labels, region_masks, _ = ppg.LinearSamplingPPG_regions_vectorized(
            corr_grid,
            k_pts=self.k_pts,
            thr=self.ppg_thr,
            image_size=self.clip_resolution,   # SAM の座標系に合わせる
            kmeans_iters=self.ppg_kmeans_iters,
        )
        # print("debug: coords[B=0][Ncls=0]", coords[0][0])
        # print("debug: coords[B=0][Ncls=1]", coords[0][1])
        return coords, labels, region_masks
    
    # 7.3 samblock処理のwrapper関数を作成
    def _run_sam_block(self, coords, labels, region_masks, image_embeddings):
        
        Ncls = coords.size(1)
        H_emb_reg, W_emb_reg = region_masks.shape[-2:]
        B, C, H_emb, W_emb = image_embeddings.shape

        # image_pe を埋め込み解像度に合わせる（元コード通り）
        image_pe = self.sam_block.prompt_encoder.get_dense_pe().to(self.device)  # (1,256,?,?)
        if image_pe.shape[-2:] != (H_emb_reg, W_emb_reg):
            image_pe = F.interpolate(image_pe, size=(H_emb_reg, W_emb_reg),
                                     mode="bilinear", align_corners=False)

        # ====== 画像ごとに MaskDecoder をチャンク推論（元コード通り） ======
        chunk = self.sam_chunk
        outs_per_img = []

        for b in range(B):
            Nk = Ncls * self.k_pts
            coords_b = coords[b].reshape(Nk, 1, 2).to(self.device)                  # (Nk,1,2)
            labels_b = labels[b].reshape(Nk, 1).to(self.device)                     # (Nk,1)
            masks_b  = region_masks[b].reshape(Nk, 1, H_emb_reg, W_emb_reg).to(self.device)

            # PromptEncoder（元コード通り）
            sparse_b, dense_b = self.sam_block.prompt_encoder(
                points=(coords_b, labels_b),
                boxes=None,
                masks=masks_b,
            )
            # SAM 内部解像度に合わせて upsample（元コード通り）
            dense_b = F.interpolate(
                dense_b, size=(self.H_emb, self.W_emb),
                mode="bilinear", align_corners=False
            )

            # SAM の Transformer をチャンクに分けて通す（B=1 で呼ぶ：元コード通り）
            src_chunks = []
            for start in range(0, Nk, chunk):
                end = min(Nk, start + chunk)
                sp = sparse_b[start:end]     # (m, 2, 256)
                de = dense_b[start:end]      # (m, 256, H, W)

                # ここで image_embeddings は「画像 b のみ」を渡す → MaskDecoder 内で m 回に自動展開
                _, src_m = self.sam_block.mask_decoder(
                    image_embeddings=image_embeddings[b:b+1],  # (1,256,H,W)
                    image_pe=image_pe,                          # (1,256,H,W)
                    sparse_prompt_embeddings=sp,                # (m,2,256)
                    dense_prompt_embeddings=de,                 # (m,256,H,W)
                    multimask_output=False,
                    sam_transformer_block_only=True,
                )
                # (m, HW, 256) → (m, 256, H, W)（元コード通り）
                m, HW, C1 = src_m.shape
                h = w = int(HW ** 0.5)
                src_m = src_m.permute(0, 2, 1).reshape(m, C1, h, w)
                src_chunks.append(src_m)

            # (Nk,256,H,W) → (Ncls,k_pts,256,H,W) → 領域集約（max）（元コード通り）(Ncls, C, H, W)
            # 1.クラスごとにFv'を生成する(Ncls, k_pts, C, H, W) -> (Ncls, C, H, W)
                # 1.1 max k_pts軸に沿って，最も高い値を取得
            # 2.一枚のF'vを生成する(Ncls, k_pts, C, H, W) -> (C, H, W)
            src_regions = torch.cat(src_chunks, dim=0).reshape(Ncls, self.k_pts, -1, self.H_emb, self.W_emb)
            
            # 4.1.2_samblock_1x1conv_forward
            # 修正前
            # src_class   = src_regions.max(dim=1).values  # (Ncls,256,H,W)
            # 修正後
            src_regions = src_regions.reshape(Ncls, -1, self.H_emb, self.W_emb) # (T, k_pts, C, H, W) -> (T, k_pts*C, H, W)
            src_class = self.sam_prompts_conv(src_regions) # (T, k_pts*C, H, W) -> (T, C, H, W)
            # ----------------
            outs_per_img.append(src_class)

        # (B,Ncls,256,H,W) → (B*Ncls,256,H,W) に畳む（元コード通り）
        src_per_class = torch.stack(outs_per_img, dim=0)
        Fv_prime = rearrange(src_per_class, "B T C H W -> B C T H W", B=B, C=C, H=H_emb)
        return Fv_prime


    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W (4, 1, 171, 24, 24)
        """

        # 6.3 ppgのwrapper関数をforwardで呼び出す
        # 修正後
        B, C, T, H, W = x.shape
        x_ppg = rearrange(x.squeeze(1), "B T H W -> B H W T", B=B, T=T, H=H)
        coords, labels, region_masks = self._run_ppg(x_ppg)
        # ----------------------

        # 1.8 corrの畳み込み処理追加
        # 修正後
        x = self.corr_embed(x)
        # ----------------------

        # 7.4 wrapper関数を用いてforwardにコード記載
        # 修正後
        # dbg(appearance_guidance=appearance_guidance)
        appearance_guidance = self._run_sam_block(
            coords=coords,
            labels=labels,
            region_masks=region_masks,
            image_embeddings=appearance_guidance
        )
        # dbg(appearance_guidance_after_sam=appearance_guidance)

        x = self.swin_block(x, appearance_guidance)
        x = self.attention(x, text_guidance)

        # 1.9 最終ESCBlockの場合は，チャンネル方向次元削除処理をスキップ
        # 修正後
        
        if self.last_block == False:
            # x = torch.mean(x, dim=1, keepdim=True)
            # dbg(x=x)
            # B, C_, T, H, W = x.shape 
            x = rearrange(x, "B C T H W -> (B T) C H W", B=B, T=T).contiguous()
            x = self.c_conv(x)
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B, T=T).contiguous()
            
        # ----------------------
        return x


class AggregatorResNetLayer(nn.Module):
    def __init__(self, hidden_dim=64, appearance_guidance=512) -> None:
        super().__init__()
        self.conv_linear = nn.Conv2d(hidden_dim + appearance_guidance, hidden_dim, kernel_size=1, stride=1)
        self.conv_layer = Bottleneck(hidden_dim, hidden_dim // 4)


    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
        """
        B, T = x.size(0), x.size(2)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        appearance_guidance = repeat(appearance_guidance, 'B C H W -> (B T) C H W', T=T)

        x = self.conv_linear(torch.cat([x, appearance_guidance], dim=1))
        x = self.conv_layer(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)


class Aggregator(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=0, # 0917_2.3_paddin_tokens無効化
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 1.2.1 AggregatorLayer初期化時に渡すprompt_channel，last_block変数を追加
        # 修正前
        """
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])
        """
        # 修正後
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
                prompt_channel=prompt_channel, last_block=True if i == num_layers -1 else False,
            ) for i in range(num_layers)
        ])
        
        # ----------------------


        # 1.4 corrの畳み込みconv2d削除
        # 修正前
        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        # ----------------------

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    # 1.3 corr畳み込み処理のwrapper関数の削除
    #　修正前
    """
    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
        """
    
    # -------------------------------
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C)
            apperance_guidance: tuple of (B, C, H, W)
        """
        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)

        # 1.1 forward内での畳み込み削除
        # 修正前
        # corr_embed = self.corr_embed(corr)
        # 修正後
        # --------------------------------

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        # 1.2 AggregatorLayer(ESCBlock)入力変数を変更
        # 修正前
        """
        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        """
        # 修正後
        for layer in self.layers:
            corr = layer(corr, projected_guidance, projected_text_guidance)
        logit = self.conv_decoder(corr, projected_decoder_guidance)
        
        # --------------------------------


        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
