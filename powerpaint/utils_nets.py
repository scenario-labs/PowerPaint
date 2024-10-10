import torch
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.downsampling import Downsample2D


def print_modules(net):
    for name, m in net.named_modules():
        if len(list(m.named_modules())) == 1:
            print(name, "\t", m)


def group_norm_affine_part(input, norm):
    # """
    # Apply the affine part of the group norm
    #
    # @param input: input tensor
    # @param norm: weight and bias will be fetched from the original group norm
    #
    # @return: normalized tensor
    # """
    assert isinstance(norm, torch.nn.GroupNorm)
    weight = norm.weight
    bias = norm.bias
    out = input
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


def resnet_step1(resnet, hidden_states):
    assert isinstance(resnet, ResnetBlock2D)
    assert resnet.upsample is None
    assert resnet.downsample is None
    assert resnet.time_emb_proj is None
    assert resnet.time_embedding_norm in [ "default", "group" ]
    hidden_states = resnet.nonlinearity(hidden_states)
    hidden_states = resnet.conv1(hidden_states)
    return hidden_states


def resnet_step2(resnet, sample):
    hidden_states = resnet.nonlinearity(sample)
    hidden_states = resnet.dropout(hidden_states)
    hidden_states = resnet.conv2(hidden_states)
    return hidden_states


def attention(attn, hidden_states):
    assert hasattr(F, "scaled_dot_product_attention") and attn.scale_qk, "hack not working outside this"
    assert attn.spatial_norm is None
    assert attn.norm_cross is None
    assert attn.residual_connection
    assert hidden_states.ndim == 4
    batch_size, channel, height, width = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, channel, height * width)
    hidden_states = hidden_states.transpose(1, 2)
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    return hidden_states


def up_sample(up_block, sample):
    if up_block.upsamplers is not None:
        for upsampler in up_block.upsamplers:
            sample = upsampler(sample)
    return sample


def down_sample(down_block, sample):
    if down_block.downsamplers is not None:
        for downsampler in down_block.downsamplers:
            assert isinstance(downsampler, Downsample2D)
            # sample = downsampler(sample)
            assert downsampler.norm is None
            #print("channels", downsampler.channels)
            assert downsampler.out_channels == downsampler.channels
            assert downsampler.use_conv
            assert downsampler.padding == 0
            assert downsampler.conv.stride == (2, 2)
            assert downsampler.conv.kernel_size == (3,3)
            if downsampler.use_conv and downsampler.padding == 0:
                pad = (0, 1, 0, 1)
                sample = F.pad(sample, pad, mode="constant", value=0)
            sample = downsampler.conv(sample)
    return sample
