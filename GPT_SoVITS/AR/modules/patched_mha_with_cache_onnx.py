from typing import Optional, Tuple
from torch.nn.functional import *
from torch.nn.functional import (
    _canonical_mask,
)


def multi_head_attention_forward_patched(
    query,
    key,
    value,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    cache=None,
) -> Tuple[Tensor, Optional[Tensor]]:
    # set up shape vars
    _, _, embed_dim = query.shape
    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )
    head_dim = embed_dim // num_heads

    proj_qkv = linear(query, in_proj_weight, in_proj_bias)
    #直接截取省去视图访问开销
    q, k, v = proj_qkv[:, :, :512], proj_qkv[:, :, 512:1024], proj_qkv[:, :, 1024:]

    stage_num = cache["stage"]
    if cache["first_infer"] == 1:
        cache["k"][:, stage_num:stage_num+1, :] = k
        cache["v"][:, stage_num:stage_num+1, :] = v
    else:
        cache["k"][-1:, stage_num:stage_num+1, :] = k
        cache["v"][-1:, stage_num:stage_num+1, :] = v
        k = cache["k"][:, stage_num:stage_num+1, :]
        v = cache["v"][:, stage_num:stage_num+1, :]
    cache["stage"] = (stage_num + 1) % cache["all_stage"]

    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )
    attn_mask = attn_mask.unsqueeze(0)

    q = q.view(-1, num_heads, head_dim).transpose(0, 1)
    k = k.view(-1, num_heads, head_dim).transpose(0, 1)
    v = v.view(-1, num_heads, head_dim).transpose(0, 1)

    dropout_p = 0.0
    attn_mask = attn_mask.unsqueeze(0)
    q = q.view(num_heads, -1, head_dim).unsqueeze(0)
    k = k.view(num_heads, -1, head_dim).unsqueeze(0)
    v = v.view(num_heads, -1, head_dim).unsqueeze(0)
    attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(-1, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(-1, 1, attn_output.size(1))

    return attn_output
