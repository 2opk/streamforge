"""
H2O P1 Simple Diagonal Caching - 가장 간단한 구현
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def H2O_p1_simple_diagonal(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, k=4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = arg0.device
    B, S, H, D = arg0.shape


    autotune_key = torch.cuda.get_device_capability(dev)[0]

    tensor_0 = arg0
    tensor_1 = arg1  # K
    tensor_2 = arg2  # V

    empty_ptr_3 = torch.empty(B, H, S, 1, dtype=torch.float32, device=dev)
    empty_ptr_4 = torch.empty(B, S, H, D, dtype=torch.float16, device=dev)

    num_blocks = S // 128  # 32
    diagonal_cache = torch.zeros(B * num_blocks, H, k, 128, 128, dtype=torch.float32, device=dev)

    grid = (1, H, S // 128)

    H2O_p1_simple_diagonal_kernel[grid](
        tensor_0, tensor_1, tensor_2,
        empty_ptr_3, empty_ptr_4, diagonal_cache,
        k, S, D, autotune_key
    )

    tensor_5 = empty_ptr_3
    tensor_6 = empty_ptr_4
    return tensor_5, tensor_6, diagonal_cache

def H2O_p1_simple_diagonal_wrapper(Q: torch.Tensor, V: torch.Tensor, K: torch.Tensor, k=4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return H2O_p1_simple_diagonal(Q, K, V, k)

@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
], key=['autotune_key'])
@triton.jit
def H2O_p1_simple_diagonal_kernel(
    arg_0, arg_1, arg_2, arg_3, arg_4, diagonal_cache,
    k: tl.constexpr, S: tl.constexpr, D: tl.constexpr, autotune_key,
):

    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    c_denom = 1.275311e-01
    c_zero = 0
    c_4096 = 4096
    c_14 = 128
    num_blocks = S // 128

    mul_15 = pid1 * c_14
    mul_16 = mul_15 * c_4096
    mul_17 = pid2 * c_14
    add_18 = mul_16 + mul_17

    block_ptr_19 = tl.make_block_ptr(
        base=arg_0 + add_18,
        shape=(128, 128,),
        strides=(4096, 1,),
        offsets=(0, 0,),
        block_shape=(128, 128,),
        order=(1, 0,),
    )
    block_load_20 = tl.load(block_ptr_19)

    mul_21 = pid2 * c_4096
    add_22 = mul_15 + mul_21
    block_ptr_23 = tl.make_block_ptr(
        base=arg_3 + add_22,
        shape=(128, 1,),
        strides=(1, 1,),
        offsets=(0, 0,),
        block_shape=(128, 1,),
        order=(1, 0,),
    )
    block_ptr_24 = tl.make_block_ptr(
        base=arg_4 + add_18,
        shape=(128, 128,),
        strides=(4096, 1,),
        offsets=(0, 0,),
        block_shape=(128, 128,),
        order=(1, 0,),
    )

    converted_25 = c_denom
    mul_26 = block_load_20 * converted_25
    mul_26 = mul_26.to(tl.float16)

    zero_27 = tl.zeros([128, 128], dtype=tl.float32)
    zero_28 = tl.zeros([128, 1], dtype=tl.float32)

    add_29 = mul_15 + c_14
    block_ptr_30 = tl.make_block_ptr(
        base=arg_1 + mul_17,  # K
        shape=(128, 4096,),
        strides=(1, 4096,),
        offsets=(0, 0,),
        block_shape=(128, 128,),
        order=(0, 1,),
    )
    block_ptr_31 = tl.make_block_ptr(
        base=arg_2 + mul_17,  # V
        shape=(4096, 128,),
        strides=(4096, 1,),
        offsets=(0, 0,),
        block_shape=(128, 128,),
        order=(1, 0,),
    )

    for i_32 in range(c_zero, add_29, c_14):
        k_block_idx = i_32 // c_14

        block_load_33 = tl.load(block_ptr_30)
        block_load_34 = tl.load(block_ptr_31)

        dot_35 = tl.dot(mul_26, block_load_33)
        where_36 = tl.zeros([128, 128], dtype=tl.float32)
        where_36 = tl.where(mul_15 + tl.arange(0, 128)[:, None] >= i_32 + tl.arange(0, 128)[None, :], where_36, float('-inf'))
        add_37 = dot_35 + where_36
        exp2_38 = tl.math.exp2(add_37)

        q_block_idx = pid2
        diagonal_offset = (k_block_idx - q_block_idx + num_blocks) % num_blocks
        is_cacheable = diagonal_offset < k

        if is_cacheable:
            cache_offset = (pid2 * k * num_blocks + pid1 * k + diagonal_offset) * 128 * 128
            cache_ptr = tl.make_block_ptr(
                base=diagonal_cache + cache_offset,
                shape=(128, 128,),
                strides=(128, 1,),
                offsets=(0, 0,),
                block_shape=(128, 128,),
                order=(1, 0,),
            )
            tl.store(cache_ptr, exp2_38.to(tl.float32))

        reduce_sum_39 = tl.sum(exp2_38, axis=1, keep_dims=True).to(tl.float32)
        zero_28 += reduce_sum_39
        converted_40 = exp2_38.to(tl.float16)
        dot_41 = tl.dot(converted_40, block_load_34)
        zero_27 += dot_41

        block_ptr_30 = tl.advance(block_ptr_30, (0, 128,))
        block_ptr_31 = tl.advance(block_ptr_31, (128, 0,))

    div_45 = zero_27 / zero_28
    converted_46 = div_45.to(tl.float16)

    tl.store(block_ptr_23, zero_28)
    tl.store(block_ptr_24, converted_46)