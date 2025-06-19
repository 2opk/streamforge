import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def H2O_p4_simple_cache(Q: torch.Tensor, K: torch.Tensor, L: torch.Tensor, diagonal_cache: torch.Tensor, k=4) -> torch.Tensor:
    dev = Q.device
    autotune_key = torch.cuda.get_device_capability(dev)[0]

    B, S, H, D = Q.shape

    output = torch.empty(B, H, S, dtype=torch.float32, device=dev)
    grid = (1, H, S // 128)

    H2O_p4_simple_cache_kernel[grid](
        Q, K, L, diagonal_cache, output,
        k, autotune_key
    )

    return output

@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
], key=['autotune_key'])
@triton.jit
def H2O_p4_simple_cache_kernel(
    arg_0, arg_1, arg_2, diagonal_cache, arg_3,
    k: tl.constexpr, autotune_key,
):
    pid_4 = tl.program_id(0)
    pid_5 = tl.program_id(1)  # head index
    pid_6 = tl.program_id(2)  # col block index

    const_7 = 1.275311e-01
    const_10 = 4096
    const_11 = 128
    num_blocks = 32  # 4096 // 128

    mul_13 = pid_6 * const_11
    mul_14 = pid_5 * const_11
    mul_15 = mul_14 * const_10
    add_16 = mul_15 + mul_13

    block_ptr_17 = tl.make_block_ptr(
        base=arg_1 + add_16,
        shape=(128, 128,),
        strides=(1, 4096,),
        offsets=(0, 0,),
        block_shape=(128, 128,),
        order=(0, 1,),
    )
    block_load_18 = tl.load(block_ptr_17)

    mul_19 = pid_6 * const_10
    add_20 = mul_14 + mul_19
    block_ptr_21 = tl.make_block_ptr(
        base=arg_3 + add_20,
        shape=(128,),
        strides=(1,),
        offsets=(0,),
        block_shape=(128,),
        order=(0,),
    )

    converted_22 = const_7
    mul_23 = block_load_18 * converted_22
    mul_23 = mul_23.to(tl.float16)

    zero_24 = tl.zeros([128], dtype=tl.float32)

    block_ptr_25 = tl.make_block_ptr(
        base=arg_0 + add_16,
        shape=(4096, 128,),
        strides=(4096, 1,),
        offsets=(0, 0,),
        block_shape=(128, 128,),
        order=(1, 0,),
    )

    block_ptr_26 = tl.make_block_ptr(
        base=arg_2 + add_20,
        shape=(4096,),
        strides=(1,),
        offsets=(0,),
        block_shape=(128,),
        order=(0,),
    )

    for i_27 in range(mul_14, const_10, const_11):
        block_load_28 = tl.load(block_ptr_25)  # Q block
        block_load_29 = tl.load(block_ptr_26)  # L block

        row_idx = i_27 // const_11
        col_idx = pid_6
        diagonal_offset = (col_idx - row_idx + num_blocks) % num_blocks
        is_cached = diagonal_offset < k

        if is_cached:
            q_block_idx = row_idx
            head_idx = pid_5
            cache_linear_offset = (q_block_idx * k * num_blocks + head_idx * k + diagonal_offset) * 128 * 128

            cache_ptr = tl.make_block_ptr(
                base=diagonal_cache + cache_linear_offset,
                shape=(128, 128,),
                strides=(128, 1,),
                offsets=(0, 0,),
                block_shape=(128, 128,),
                order=(1, 0,),
            )
            exp2_33 = tl.load(cache_ptr)

        else:
            where_30 = tl.zeros([128, 128], dtype=tl.float32)
            where_30 = tl.where(
                i_27 + tl.arange(0, 128)[:, None] >= mul_14 + tl.arange(0, 128)[None, :],
                where_30,
                float('-inf')
            )
            dot_31 = tl.dot(block_load_28, mul_23)
            add_32 = dot_31 + where_30
            exp2_33 = tl.math.exp2(add_32)

        unsqueeze_34 = block_load_29[:, None]
        div_35 = exp2_33 / unsqueeze_34
        reduce_sum_36 = tl.sum(div_35, axis=0, keep_dims=False).to(tl.float32)
        reduce_sum_36 += zero_24

        block_advance_37 = tl.advance(block_ptr_25, (128, 0,))
        block_advance_38 = tl.advance(block_ptr_26, (128,))
        block_ptr_25 = block_advance_37
        block_ptr_26 = block_advance_38
        zero_24 = reduce_sum_36
    block_store_39 = tl.store(block_ptr_21, zero_24)

def H2O_p4_simple_cache_wrapper(Q: torch.Tensor, K: torch.Tensor, L: torch.Tensor, diagonal_cache: torch.Tensor, k=4) -> torch.Tensor:
    return H2O_p4_simple_cache(Q, K, L, diagonal_cache, k)