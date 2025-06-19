import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

# --- Benchmarking Functions ---

def bench_H2O_part1():
  dev = torch.cuda.current_device()
  # Based on H2O call: H2O_p1(query_tensor, value_tensor, key_tensor)
  # So, arg0 for H2O_p1 is query, arg1 is value, arg2 is key
  sample_query_tensor = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  sample_value_tensor = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev) # This will be arg1 for H2O_p1
  sample_key_tensor = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)   # This will be arg2 for H2O_p1

  avg_ms = triton.testing.do_bench(lambda: H2O_part1_forward_attention(sample_query_tensor, sample_value_tensor, sample_key_tensor))
  print('[H2O_part1_forward_attention] avg_ms:', avg_ms)

def bench_H2O_part2_grad_query():
  dev = torch.cuda.current_device()
  # Based on H2O call: H2O_p4(weighted_value_buffer, query_tensor, sum_exp_buffer)
  # arg0 for H2O_p4 is weighted_value_buffer (output_o from p1)
  # arg1 for H2O_p4 is query_tensor
  # arg2 for H2O_p4 is sum_exp_buffer (log_sum_exp from p1)
  sample_fwd_output_o_tensor = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  sample_query_tensor = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  sample_sum_exp_tensor = torch.randn(1, 32, 4096, 1, dtype=torch.float32, device=dev)

  avg_ms = triton.testing.do_bench(lambda: H2O_part2_backward_calc_grad_query(sample_fwd_output_o_tensor, sample_query_tensor, sample_sum_exp_tensor))
  print('[H2O_part2_backward_calc_grad_query] avg_ms:', avg_ms)

# --- H2O Main Function ---

def H2O_attention_mechanism(
    query_tensor: torch.Tensor,
    key_tensor: torch.Tensor,
    value_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Performs the H2O attention mechanism.
  H2O_part1 computes attention output and sum of exponents (for stable softmax).
  H2O_part2 seems to compute a gradient component related to the query, using outputs from part1.
  """
  # In H2O_p1 call from original H2O: arg_0 (query), arg_2 (value), arg_1 (key)
  # So H2O_part1_forward_attention(query_tensor, value_tensor, key_tensor)
  # k0_out_0 -> sum_exp_buffer, k0_out_1 -> weighted_value_buffer
  sum_exp_buffer, weighted_value_buffer = H2O_part1_forward_attention(query_tensor, value_tensor, key_tensor)

  # In H2O_p4 call from original H2O: k0_out_1 (weighted_value_buffer), arg_0 (query_tensor), k0_out_0 (sum_exp_buffer)
  # k1_out_0 -> grad_query_buffer
  grad_query_buffer = H2O_part2_backward_calc_grad_query(weighted_value_buffer, query_tensor, sum_exp_buffer)

  return weighted_value_buffer, grad_query_buffer

# --- H2O Part 1: Forward Attention Computation ---

def H2O_part1_forward_attention(
    query_input: torch.Tensor,    # Original arg0
    value_input: torch.Tensor,    # Original arg1 (which was arg2 in H2O func's perspective)
    key_input: torch.Tensor       # Original arg2 (which was arg1 in H2O func's perspective)
) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = query_input.device
  autotune_key_capability = torch.cuda.get_device_capability(dev)[0]

  # Output buffers
  # sum_exp_output_buffer shape (1, 32, 4096, 1) -> (Batch, Heads, SeqLen_Q, 1)
  sum_exp_output_buffer = torch.empty(1, 32, query_input.shape[1], 1, dtype=torch.float32, device=dev)
  # weighted_value_output_buffer shape (1, 4096, 32, 128) -> (Batch, SeqLen_Q, Heads, HeadDim)
  weighted_value_output_buffer = torch.empty_like(query_input, dtype=torch.float16) # Same shape as query_input

  grid_dim_z = 1 # Corresponds to batch or other high-level grouping
  grid_dim_h = query_input.shape[2] # Number of heads
  # Assuming query_input.shape[1] is SeqLen_Q and 128 is BLOCK_SIZE_M for query blocks
  grid_dim_m_blocks = query_input.shape[1] // 128 # Number of query blocks over sequence length

  grid = (grid_dim_z, grid_dim_h, grid_dim_m_blocks)

  H2O_part1_kernel_attention_fwd[grid](
      query_input,
      value_input,
      key_input,
      sum_exp_output_buffer,
      weighted_value_output_buffer,
      autotune_key_capability
  )
  return sum_exp_output_buffer, weighted_value_output_buffer

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key_capability'])
@triton.jit
def H2O_part1_kernel_attention_fwd(
    query_ptr,       # Input Q: (B, N_q, H, D_k)
    value_ptr,       # Input V: (B, N_kv, H, D_v)
    key_ptr,         # Input K: (B, N_kv, H, D_k)
    sum_exp_out_ptr, # Output L (logsumexp): (B, H, N_q, 1)
    output_o_ptr,    # Output O: (B, N_q, H, D_v)
    autotune_key_capability, # For autotuning based on CUDA capability
):
  # Program IDs
  z_idx = tl.program_id(0) # Batch/group index (unused if grid_dim_z=1)
  h_idx = tl.program_id(1) # Head index
  m_block_idx = tl.program_id(2) # Query sequence block index (for M dimension)

  # Constants
  KERNEL_SCALE_FACTOR = 1.275311e-01 # Scaling factor for Q*K
  NEGATIVE_INFINITY = float('-inf')
  ZERO_FP32 = 0.0
  ZERO_INT = 0
  # ONE_INT = 1 # Not used directly

  # Assuming dimensions from typical attention.
  # query_ptr shape (1, 4096, 32, 128) -> (B, N_q, H, D_k=D_v)
  # value_ptr shape (1, 4096, 32, 128) -> (B, N_kv, H, D_v)
  # key_ptr shape (1, 4096, 32, 128)   -> (B, N_kv, H, D_k)
  # These are logical shapes. Triton accesses them based on strides.

  SEQ_LEN_KV = tl.constexpr(key_ptr.shape[1]) # Typically N_kv, here 4096
  HEAD_DIM = tl.constexpr(query_ptr.shape[3]) # D_k or D_v, here 128

  BLOCK_SIZE_M = tl.constexpr(128) # Block size for query sequence dimension
  BLOCK_SIZE_N = tl.constexpr(128) # Block size for key/value sequence dimension
  # Note: HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N are all 128 in this specific kernel.

  # --- Pointer Offsets for Q, O ---
  # Offset for the current head and query block.
  # Assumes data layout (B, H, N, D) or needs specific striding.
  # Original: mul_15 = h_idx * 128; mul_16 = mul_15 * 4096; mul_17 = m_block_idx * 128; add_18 = mul_16 + mul_17
  # This offset structure (h_idx * D * N_q + m_block_idx * D) is for (H, M_block, D_k_block, D_k_elem) like access
  # Or for (B,H,N,D) if strides are set up for it.
  # Let's check make_block_ptr for query_ptr (arg_0)
  # base=query_ptr + (h_idx * HEAD_DIM * SEQ_LEN_Q) + (m_block_idx * HEAD_DIM)
  # strides=(SEQ_LEN_Q, 1) for a block_shape=(BLOCK_SIZE_M, HEAD_DIM) implies Q is (M_dim_elements, K_dim_elements)
  # This is a Q block of M_rows x K_cols.
  # query_ptr has shape (1, 4096, 32, 128) -> (B, N_q, H, D_k)
  # The kernel seems to be processing data as if Q is (B, H, N_q, D_k) after transposition/view.
  # offset for Q: (batch_idx * stride_b) + (h_idx * stride_h) + (m_block_idx * BLOCK_SIZE_M * stride_n_q_element)
  # For query_ptr (Q) of shape (1, N_q, H, D_k=128)
  # Effective offset for query block at (h_idx, m_block_idx * BLOCK_SIZE_M) for a specific batch (0 here)
  q_block_start_row_offset = m_block_idx * BLOCK_SIZE_M
  q_ptr_base_offset = h_idx * HEAD_DIM + q_block_start_row_offset * query_ptr.numel // (query_ptr.shape[0] * query_ptr.shape[2]) # approximate, depends on actual strides from PyTorch
  # The original `add_18` is (h_idx * 128 * 4096) + (m_block_idx * 128) using element count.
  # If Q tensor is (B, N_q, H, D_k), its default strides are (N_q*H*D_k, H*D_k, D_k, 1).
  # The kernel implicitly uses a view/transpose: (B, H, N_q, D_k). Strides: (H*N_q*D_k, N_q*D_k, D_k, 1).
  # Offset for head `h_idx`: h_idx * N_q * D_k
  # Offset for m_block: m_block_idx * BLOCK_SIZE_M * D_k
  # Offset for q_block_ptr: (h_idx * SEQ_LEN_KV * HEAD_DIM) (using SEQ_LEN_KV from original kernel for strides) + (m_block_idx * BLOCK_SIZE_M) for element stride=1
  # Let's use the original calculation for add_18 for safety.
  _h_offset_intermediate = h_idx * HEAD_DIM # const_14 = HEAD_DIM
  _q_o_base_offset_for_head = _h_offset_intermediate * SEQ_LEN_KV # const_13 = SEQ_LEN_KV (originally 4096)
  _q_o_base_offset_for_m_block = m_block_idx * HEAD_DIM
  q_o_start_ptr_offset = _q_o_base_offset_for_head + _q_o_base_offset_for_m_block

  q_block_ptr = tl.make_block_ptr(
    base=query_ptr + q_o_start_ptr_offset, # Pointer to Q data for the current head and m_block
    shape=(BLOCK_SIZE_M, HEAD_DIM),        # Shape of the Q block (e.g., 128x128)
    strides=(SEQ_LEN_KV, 1),               # Strides to traverse Q data (original: (4096,1))
    offsets=(0, 0),                        # Start offsets within the block
    block_shape=(BLOCK_SIZE_M, HEAD_DIM),  # Meta-shape of the block
    order=(1, 0)                           # Memory layout order
  )
  q_block_data = tl.load(q_block_ptr) # Load Q_mk (M rows, K_dim columns)

  # --- Pointer Offsets for L (sum_exp_out_ptr) ---
  # sum_exp_out_ptr shape (1, 32, 4096, 1) -> (B, H, N_q, 1)
  # Offset for (h_idx, m_block_idx * BLOCK_SIZE_M)
  # Original: mul_21 = m_block_idx * 4096; add_22 = (h_idx * 128) + mul_21
  # This suggests (h_idx * BLOCK_SIZE_M) + (m_block_idx * SEQ_LEN_Q) for (H, M) structure
  l_ptr_offset = (h_idx * BLOCK_SIZE_M) + (m_block_idx * SEQ_LEN_KV) # SEQ_LEN_KV was 4096

  sum_exp_block_ptr = tl.make_block_ptr(
    base=sum_exp_out_ptr + l_ptr_offset, # Pointer to L data for current head and m_block
    shape=(BLOCK_SIZE_M, 1),             # Shape of L block (M rows, 1 column)
    strides=(1, 1),                      # Strides for L (original: (1,1))
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_M, 1),
    order=(1, 0) # or (0,) if 1D
  )

  # --- Pointer Offsets for O (output_o_ptr) ---
  # output_o_ptr has same shape and assumed layout as query_ptr
  output_o_block_ptr = tl.make_block_ptr(
    base=output_o_ptr + q_o_start_ptr_offset, # Pointer to O data, similar to Q
    shape=(BLOCK_SIZE_M, HEAD_DIM),           # Shape of O block (M rows, V_dim columns)
    strides=(SEQ_LEN_KV, 1),                  # Strides for O (original: (4096,1))
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_M, HEAD_DIM),
    order=(1, 0)
  )

  # Scale Q block
  scaled_q_block_data = (q_block_data * KERNEL_SCALE_FACTOR).to(tl.float16) # Q_mk_scaled

  # Accumulators
  output_accumulator = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32) # Accumulates O_mv parts
  sum_exp_accumulator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)       # Accumulates L_m parts (sum of exp scores)

  # --- Pointer Offsets for K, V ---
  # Original: mul_17 = m_block_idx * 128
  # This m_block_idx is for the *query* block. For K/V, the iteration variable i_32 (n_block_offset) is used.
  # The base for K/V in the loop is just key_ptr / value_ptr + head_offset + n_block_offset
  # The fixed part related to head for K/V:
  # For K/V of shape (B, N_kv, H, D), viewed as (B, H, N_kv, D)
  # Effective head offset for K/V: h_idx * N_kv * D
  # This is complex due to how Triton handles N-D tensors. The original fixed mul_17 implies fixed offset for all N blocks
  # which is not standard attention. Let's assume mul_17 refers to m_block_idx of Query.
  # base_kv_offset_for_m_query_block = m_block_idx * HEAD_DIM # Original mul_17
  # This offset is unusual for K/V if m_block_idx is for query blocks.
  # Let's assume this specific H2O kernel has K and V blocks aligned with Q blocks in some way via `m_block_idx * HEAD_DIM`
  # or that `key_ptr` and `value_ptr` are already pre-offset.
  # More typical: K/V base depends on h_idx, and then n_block_offset in the loop.
  # For simplicity and adherence, we use the original's `mul_17` which uses `m_block_idx` for base K,V ptrs.
  kv_base_offset_from_m_block = m_block_idx * HEAD_DIM

  # Loop over Key/Value sequence blocks (N dimension)
  # Original: add_29 = (h_idx + 1) * 128. loop_upper_bound_n
  # This upper bound depends on h_idx, which is specific to H2O algorithm.
  loop_n_upper_bound = (h_idx + 1) * BLOCK_SIZE_N

  for n_block_start_offset in range(ZERO_INT, loop_n_upper_bound, BLOCK_SIZE_N):
    # K block pointer: K_kn (K_dim rows, N_block columns)
    k_block_ptr = tl.make_block_ptr(
      base=key_ptr + kv_base_offset_from_m_block, # Offset based on m_block_idx (query block)
      shape=(HEAD_DIM, SEQ_LEN_KV),             # Shape of full K for this head (Dk, Nkv)
      strides=(1, SEQ_LEN_KV),                  # Strides (original: (1, 4096)) - this is unusual, usually (N_kv, 1) or (D_k_stride, N_kv_stride)
                                                # (1, SEQ_LEN_KV) for shape (HEAD_DIM, SEQ_LEN_KV) means K is stored Dk-major for a given N_kv block.
                                                # The original had strides (1, 4096) for a (128,4096) shape, and (128,128) block_shape, order(0,1).
                                                # This means K_block[col,row] according to (0,1) order.
      offsets=(0, n_block_start_offset),        # Offset to current N block
      block_shape=(HEAD_DIM, BLOCK_SIZE_N),     # K_block is (Dk, N_block_size)
      order=(0, 1)                              # Order (0,1) means tl.dot views it as K_dim x N_block
    )
    k_block_data = tl.load(k_block_ptr) # Load K_kn_block (D_k rows, N_block_cols)

    # V block pointer: V_nv (N_block rows, V_dim columns)
    v_block_ptr = tl.make_block_ptr(
      base=value_ptr + kv_base_offset_from_m_block, # Offset based on m_block_idx (query block)
      shape=(SEQ_LEN_KV, HEAD_DIM),             # Shape of full V for this head (Nkv, Dv)
      strides=(SEQ_LEN_KV, 1),                  # Strides (original: (4096,1))
      offsets=(n_block_start_offset, 0),        # Offset to current N block
      block_shape=(BLOCK_SIZE_N, HEAD_DIM),     # V_block is (N_block_size, Dv)
      order=(1, 0)                              # Order (1,0) means tl.dot views it as N_block x V_dim
    )
    v_block_data = tl.load(v_block_ptr) # Load V_nv_block (N_block_rows, D_v_cols)

    # Compute S_mn_block = Q_mk_scaled @ K_kn_block.T (effectively K_nk)
    # scaled_q_block_data is (BLOCK_SIZE_M, HEAD_DIM)
    # k_block_data is (HEAD_DIM, BLOCK_SIZE_N) due to order (0,1) for dot
    scores_block = tl.dot(scaled_q_block_data, k_block_data) # Result is (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Apply causal mask (specific H2O mask logic)
    # Original where: mul_15 + tl.arange(0, 128)[:, None] >= i_32 + tl.arange(0, 128)[None, :]
    # mul_15 = h_idx * BLOCK_SIZE_M
    # i_32 = n_block_start_offset
    m_indices = tl.arange(0, BLOCK_SIZE_M)[:, None]
    n_indices = tl.arange(0, BLOCK_SIZE_N)[None, :]

    # This mask is complex and specific to H2O
    # It seems to compare absolute positions based on head index and current M/N block positions
    mask_condition_lhs = (h_idx * BLOCK_SIZE_M) + m_indices
    mask_condition_rhs = n_block_start_offset + n_indices

    causal_mask = mask_condition_lhs >= mask_condition_rhs
    masked_scores_block = tl.where(causal_mask, scores_block, NEGATIVE_INFINITY)

    # Compute P_mn_block = exp2(S_mn_block_masked)
    # Using exp2 as in original kernel
    probs_block_unnormalized = tl.math.exp2(masked_scores_block.to(tl.float32)) # (M, N)

    # Update sum_exp_accumulator (L_m)
    current_block_sum_exp = tl.sum(probs_block_unnormalized, axis=1, keep_dims=True) # Sum over N dim, result (M, 1)
    sum_exp_accumulator += current_block_sum_exp

    # Update output_accumulator (O_mv)
    # O_mv += P_mn_block @ V_nv_block
    # probs_block_unnormalized is (M,N), v_block_data is (N,V_dim)
    output_contribution = tl.dot(probs_block_unnormalized.to(tl.float16), v_block_data) # Result (M, V_dim)
    output_accumulator += output_contribution

  # Normalize output: O_mv = O_mv / L_m
  final_output_block = output_accumulator / sum_exp_accumulator
  final_output_block = final_output_block.to(tl.float16)

  # Store results
  tl.store(sum_exp_block_ptr, sum_exp_accumulator) # Store L_m
  tl.store(output_o_block_ptr, final_output_block)   # Store O_mv

# --- H2O Part 2: Backward pass component (calculating dQ-like term) ---

def H2O_part2_backward_calc_grad_query(
    fwd_output_o_tensor: torch.Tensor, # Original arg0, (k0_out_1 from p1) -> weighted_value_buffer
    query_q_tensor: torch.Tensor,      # Original arg1, (arg_0 from H2O) -> original query
    sum_exp_l_tensor: torch.Tensor     # Original arg2, (k0_out_0 from p1) -> sum_exp_buffer
) -> torch.Tensor:
  dev = fwd_output_o_tensor.device
  autotune_key_capability = torch.cuda.get_device_capability(dev)[0]

  # grad_query_dq_output_buffer shape (1, 32, 4096) -> (B, H, SeqLen_Q)
  # Note: last dim is different from other tensors. It's a reduction.
  grad_query_dq_output_buffer = torch.empty(sum_exp_l_tensor.shape[0], sum_exp_l_tensor.shape[1], sum_exp_l_tensor.shape[2], dtype=torch.float32, device=dev)
  # Original empty_ptr_3 was (1, 32, 4096), so (B,H,N). The sum_exp_l_tensor has shape (1,32,4096,1). We use its first 3 dims.

  grid_dim_z = 1
  grid_dim_h = query_q_tensor.shape[2] # Number of heads
  grid_dim_m_blocks = query_q_tensor.shape[1] // 128 # Number of query blocks

  grid = (grid_dim_z, grid_dim_h, grid_dim_m_blocks)

  H2O_part2_kernel_grad_q[grid](
      fwd_output_o_tensor,
      query_q_tensor,
      sum_exp_l_tensor,
      grad_query_dq_output_buffer,
      autotune_key_capability
  )
  return grad_query_dq_output_buffer

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key_capability'])
@triton.jit
def H2O_part2_kernel_grad_q(
    # Inputs from forward pass or original inputs
    fwd_o_ptr,    # Input O (output of H2O_part1): (B, N_q, H, D_v)
    query_q_ptr,  # Input Q (original query): (B, N_q, H, D_k)
    sum_exp_l_ptr,# Input L (sum_exp from H2O_part1): (B, H, N_q, 1)
    # Output
    grad_q_dq_output_ptr, # Output dQ (gradient for Q): (B, H, N_q)
    autotune_key_capability,
):
  # Program IDs
  z_idx = tl.program_id(0) # Batch/group index
  h_idx = tl.program_id(1) # Head index
  m_block_idx = tl.program_id(2) # Query sequence block index (M dimension)

  # Constants
  KERNEL_SCALE_FACTOR = 1.275311e-01
  # NEGATIVE_INFINITY = float('-inf') # Not used directly in mask calculation here by value
  # ZERO_FP32 = 0.0                   # Not used directly

  SEQ_LEN_N = tl.constexpr(fwd_o_ptr.shape[1]) # Sequence length, typically N_q or N_kv (4096)
  HEAD_DIM = tl.constexpr(query_q_ptr.shape[3])  # D_k or D_v (128)

  BLOCK_SIZE_M = tl.constexpr(128) # Block size for M dimension (query sequence)
  BLOCK_SIZE_N = tl.constexpr(128) # Block size for N dimension (key/value sequence, or iteration dim)

  # --- Pointer Offsets for Q, O ---
  # Original: mul_13 = m_block_idx * 128; mul_14 = h_idx * 128;
  #           mul_15 = mul_14 * 4096; add_16 = mul_15 + mul_13
  # This is similar to q_o_start_ptr_offset in H2O_part1_kernel
  _h_offset_intermediate = h_idx * HEAD_DIM
  _q_o_base_offset_for_head = _h_offset_intermediate * SEQ_LEN_N
  _q_o_base_offset_for_m_block = m_block_idx * HEAD_DIM
  q_o_start_ptr_offset = _q_o_base_offset_for_head + _q_o_base_offset_for_m_block

  # Q block pointer (Query)
  # Original strides (1, 4096), order (0,1) for block (128,128) from query_q_ptr (B,N,H,Dk)
  # This loads a block Q_nk (N elements from seq dim, K elements from head_dim)
  # (effectively Dk x M_block if M is the seq dim being blocked by m_block_idx)
  q_block_ptr = tl.make_block_ptr(
    base=query_q_ptr + q_o_start_ptr_offset,
    shape=(HEAD_DIM, SEQ_LEN_N),         # Shape (Dk, Nq) for this head
    strides=(1, HEAD_DIM),               # Strides (original: (1,4096)). (1, Dk_stride_in_N_direction)
                                         # Original (1,4096) for shape (128,128) means row_stride=4096, col_stride=1.
                                         # Order (0,1) makes it effectively K_dim x M_slice
    offsets=(0, m_block_idx * BLOCK_SIZE_M), # offset for m_block_idx in N dim of (Dk,Nq) view
    block_shape=(HEAD_DIM, BLOCK_SIZE_M),# Q block is Dk x M_block_size
    order=(0, 1)                         # tl.dot sees it as (Dk, M_block)
  )
  q_block_data = tl.load(q_block_ptr)
  scaled_q_block_data = (q_block_data * KERNEL_SCALE_FACTOR).to(tl.float16) # Scaled Q (Dk, M_block)

  # --- Pointer Offsets for dQ output and L input ---
  # Original: mul_19 = m_block_idx * 4096; add_20 = (h_idx * 128) + mul_19
  # This is similar to l_ptr_offset in H2O_part1_kernel
  dq_l_ptr_offset = (h_idx * BLOCK_SIZE_M) + (m_block_idx * SEQ_LEN_N) # (H, M) structure

  # dQ output block pointer (for one M block)
  # grad_q_dq_output_ptr is (B, H, N_q)
  dq_output_block_ptr = tl.make_block_ptr(
    base=grad_q_dq_output_ptr + dq_l_ptr_offset,
    shape=(BLOCK_SIZE_M,),              # dQ_m block has M elements
    strides=(1,),
    offsets=(0,),
    block_shape=(BLOCK_SIZE_M,),
    order=(0,)
  )

  # Accumulator for dQ_m block
  dq_m_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

  # Loop over N dimension (iterating in blocks of BLOCK_SIZE_N)
  # Original loop: for i_27 in range(mul_14, const_10, const_11)
  # mul_14 = h_idx * BLOCK_SIZE_N (const_11)
  # const_10 = SEQ_LEN_N
  # loop_n_start = h_idx * BLOCK_SIZE_N # This makes loop range dependent on head index
  # loop_n_upper = SEQ_LEN_N

  # The loop in the original code: range(h_idx * BLOCK_SIZE_N, SEQ_LEN_N, BLOCK_SIZE_N)
  # This is a very specific iteration pattern.
  n_iteration_start = h_idx * BLOCK_SIZE_N

  for n_block_iter_offset in range(n_iteration_start, SEQ_LEN_N, BLOCK_SIZE_N):
    # O block pointer (Forward Output O)
    # Original strides (4096,1), order (1,0) for block (128,128) from fwd_o_ptr (B,N,H,Dv)
    # This loads a block O_nd (N elements from seq dim, D elements from head_dim)
    o_block_ptr = tl.make_block_ptr(
      base=fwd_o_ptr + q_o_start_ptr_offset, # Same base offset calculation as Q
      shape=(SEQ_LEN_N, HEAD_DIM),         # Shape (Nq, Dv) for this head
      strides=(HEAD_DIM, 1),               # Strides (original: (4096,1)). (Dv_stride_in_N_direction, 1)
                                           # Original (4096,1) for shape (128,128) means row_stride=1, col_stride=4096 (if order was 0,1)
                                           # With order (1,0) it's effectively N_slice x V_dim
      offsets=(n_block_iter_offset, 0),    # Offset for current N block iteration
      block_shape=(BLOCK_SIZE_N, HEAD_DIM),# O block is N_block_size x Dv
      order=(1, 0)                         # tl.dot sees it as (N_block, V_dim)
    )
    o_block_data_n = tl.load(o_block_ptr) # O_nd block (N_iter_block, D_v)

    # L block pointer (Sum Exp L)
    # sum_exp_l_ptr is (B,H,N_q,1). We need L_n corresponding to O_nd.
    l_block_ptr = tl.make_block_ptr(
      base=sum_exp_l_ptr + dq_l_ptr_offset, # Base offset for (h, m_block)
      shape=(SEQ_LEN_N,),                  # L for this head, m_block (N_q elements)
                                           # sum_exp_l_ptr has last dim 1, so stride for N_q is 1.
      strides=(1,),                        # Stride along N_q dimension
      offsets=(n_block_iter_offset,),      # Offset for current N_block_iter
      block_shape=(BLOCK_SIZE_N,),         # L_n block (N_iter_block elements)
      order=(0,)
    )
    l_data_n = tl.load(l_block_ptr) # L_n block (N_iter_block,)

    # Mask (H2O specific logic)
    # Original where: i_27 + tl.arange(0, 128)[:, None] >= mul_14 + tl.arange(0, 128)[None, :]
    # i_27 = n_block_iter_offset
    # mul_14 = h_idx * BLOCK_SIZE_M (this was BLOCK_SIZE_N in loop range)
    # Let's assume mul_14 (h_idx * BLOCK_SIZE) consistently refers to the same BLOCK_SIZE for the mask.
    # If BLOCK_SIZE_M = BLOCK_SIZE_N = 128.
    n_indices_iter_mask = tl.arange(0, BLOCK_SIZE_N)[:, None] # Current N block indices (rows)
    m_indices_mask = tl.arange(0, BLOCK_SIZE_M)[None, :]      # Current M block indices (cols)

    mask_condition_lhs_p2 = n_block_iter_offset + n_indices_iter_mask
    mask_condition_rhs_p2 = (h_idx * BLOCK_SIZE_M) + m_indices_mask # Assuming BLOCK_SIZE_M here

    attention_mask_p2 = mask_condition_lhs_p2 >= mask_condition_rhs_p2 # (N_iter_block, M_block)

    # Compute O @ Q_scaled.T = term_similar_to_scores
    # o_block_data_n is (N_iter_block, D_v), assume D_v = D_k
    # scaled_q_block_data is (D_k, M_block)
    # Result should be (N_iter_block, M_block)
    o_q_dot_term = tl.dot(o_block_data_n, scaled_q_block_data) # (N_iter_block, M_block)

    # Add mask (element-wise, relies on broadcasting if NEGATIVE_INFINITY was used)
    # Original used 0 and NEGATIVE_INFINITY. Here the mask is boolean.
    # The original `where_30 = tl.zeros...; where_30 = tl.where(cond, where_30, neg_inf); add_32 = dot + where_30`
    # This implies the mask should be added (0 for pass, -inf for block).
    masked_o_q_dot_term = tl.where(attention_mask_p2, o_q_dot_term, float('-inf'))

    # exp2 of the term
    exp2_o_q_dot_term = tl.math.exp2(masked_o_q_dot_term.to(tl.float32)) # (N_iter_block, M_block)

    # Divide by L_n (reshaped)
    # l_data_n is (N_iter_block,), needs to be (N_iter_block, 1) for broadcasting
    l_data_n_reshaped = l_data_n[:, None]
    term_divided_by_l = exp2_o_q_dot_term / l_data_n_reshaped # (N_iter_block, M_block)

    # Sum over N_iter_block dimension to get dQ_m contribution
    # axis=0 sums over rows (N_iter_block dimension)
    dq_m_contribution = tl.sum(term_divided_by_l, axis=0, keep_dims=False) # Result (M_block,)

    dq_m_accumulator += dq_m_contribution.to(tl.float32)

  # Store accumulated dQ_m block
  tl.store(dq_output_block_ptr, dq_m_accumulator)