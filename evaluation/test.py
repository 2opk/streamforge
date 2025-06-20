import os, sys
import click
import torch
import numpy as np
import random

from asuka_exp.cases.kernels import KERNEL_ZOO
from asuka_exp.utils import perf, compare, display
from compile import compile

def setup_test_environment():
    """Set up test environment with fixed parameters"""
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model_name = 'h2o'
    seqlen = 4096
    cls = KERNEL_ZOO[model_name]
    model = cls()
    model = model.eval().cuda()
    specs = model.prepare(q_len=seqlen, kv_len=seqlen)
    input_names = list(specs['input'].keys())
    inputs = [specs['input'][name] for name in input_names]
    output_names = specs['output']

    print(f"Model: {model_name}, SeqLen: {seqlen}")
    print(f"Input shapes: {[inp.shape for inp in inputs]}")

    return model, inputs, input_names, output_names

def test_standard_systems(system):
    """Test torch, torchinductor, tensorrt systems"""
    print(f"\n=== Testing {system.upper()} System ===")

    model, inputs, input_names, output_names = setup_test_environment()
    f = compile(
        model=model,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        system=system,
    )
    torch.cuda.synchronize()
    outs_ref = model(*inputs)
    torch.cuda.synchronize()
    current_outs = f(*inputs)
    compare(current_outs, outs_ref, output_names)
    perf(
        label=system,
        f=f,
        args=inputs,
        run=10,
        warmup=3,
        profile=True,
    )

def test_flashtensor_system():
    """Test FlashTensor system with individual P1/P4 latency reporting"""
    print(f"\n=== Testing FLASHTENSOR System ===")

    model, inputs, input_names, output_names = setup_test_environment()
    from kernel.h2o import H2O_p1, H2O_p4, H2O
    torch.cuda.synchronize()
    outs_ref = model(*inputs)
    for _ in range(3):
        _ = H2O_p1(inputs[0], inputs[2], inputs[1])
        _ = H2O_p4(inputs[0], inputs[1], _[0])
        _ = H2O(*inputs)

    print("P1 kernel...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_time.record()
    for _ in range(10):
        p1_L, p1_O = H2O_p1(inputs[0], inputs[2], inputs[1])
    end_time.record()
    torch.cuda.synchronize()
    p1_latency = start_time.elapsed_time(end_time) / 10

    print("P4 kernel...")
    torch.cuda.synchronize()
    start_time.record()
    for _ in range(10):
        p4_out = H2O_p4(inputs[0], inputs[1], p1_L)
    end_time.record()
    torch.cuda.synchronize()
    p4_latency = start_time.elapsed_time(end_time) / 10
    torch.cuda.synchronize()
    start_time.record()
    for _ in range(10):
        ft_outs = H2O(*inputs)
    end_time.record()
    torch.cuda.synchronize()
    complete_latency = start_time.elapsed_time(end_time) / 10
    print(f"  P1 kernel latency: {p1_latency:.3f} ms")
    print(f"  P4 kernel latency: {p4_latency:.3f} ms")
    print(f"  Total latency:     {complete_latency:.3f} ms")
    compare(ft_outs, outs_ref, output_names)

def test_ours_system(diagonal_k):
    """Test our optimized system with diagonal caching"""
    print(f"\n=== Testing OURS System (diagonal_k={diagonal_k}) ===")

    model, inputs, input_names, output_names = setup_test_environment()
    from kernel.h2o_p1_simple_diagonal import H2O_p1_simple_diagonal_wrapper
    from kernel.h2o_p4_simple_cache import H2O_p4_simple_cache_wrapper
    torch.cuda.synchronize()
    outs_ref = model(*inputs)
    for _ in range(3):
        p1_L, p1_O, diagonal_cache = H2O_p1_simple_diagonal_wrapper(*inputs, k=diagonal_k)
        _ = H2O_p4_simple_cache_wrapper(inputs[0], inputs[1], p1_L, diagonal_cache, k=diagonal_k)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_time.record()
    for _ in range(10):
        p1_L, p1_O, diagonal_cache = H2O_p1_simple_diagonal_wrapper(*inputs, k=diagonal_k)
    end_time.record()
    torch.cuda.synchronize()
    p1_latency = start_time.elapsed_time(end_time) / 10

    torch.cuda.synchronize()
    start_time.record()
    for _ in range(10):
        p4_out = H2O_p4_simple_cache_wrapper(inputs[0], inputs[1], p1_L, diagonal_cache, k=diagonal_k)
    end_time.record()
    torch.cuda.synchronize()
    p4_latency = start_time.elapsed_time(end_time) / 10
    torch.cuda.synchronize()
    start_time.record()
    for _ in range(10):
        p1_L, p1_O, diagonal_cache = H2O_p1_simple_diagonal_wrapper(*inputs, k=diagonal_k)
        p4_out = H2O_p4_simple_cache_wrapper(inputs[0], inputs[1], p1_L, diagonal_cache, k=diagonal_k)
    end_time.record()
    torch.cuda.synchronize()
    complete_latency = start_time.elapsed_time(end_time) / 10
    print("OURS Performance Results:")
    print(f"  P1 kernel latency: {p1_latency:.3f} ms")
    print(f"  P4 kernel latency: {p4_latency:.3f} ms")
    print(f"  Total latency:     {complete_latency:.3f} ms")

@click.command()
@click.option('--system', '-s', required=True, help='System name: torch, torchinductor, tensorrt, flashtensor, ours')
@click.option('--diagonal_k', default=1, help='Diagonal cache parameter for ours system (1-31)')
def main(system, diagonal_k):
    print(f"System: {system}, diagonal_k: {diagonal_k}")
    valid_systems = ['torch', 'torchinductor', 'tensorrt', 'flashtensor', 'ours']
    if system not in valid_systems:
        print(f"Invalid system: {system}")
        print(f"Valid systems: {valid_systems}")
        return
    if system == 'ours':
        if not (1 <= diagonal_k <= 31):
            print(f"diagonal_k must be between 1 and 31, got: {diagonal_k}")
            return

    try:
        if system in ['torch', 'torchinductor', 'tensorrt']:
            test_standard_systems(system)
        elif system == 'flashtensor':
            test_flashtensor_system()
        elif system == 'ours':
            test_ours_system(diagonal_k)

        print(f"\n{system.upper()} testing completed successfully!")

    except Exception as e:
        print(f"\n❌ Error testing {system}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()