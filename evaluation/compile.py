import torch
import numpy as np
import onnx
from asuka_exp.utils import torch_module_to_onnx

def compile(model, input_names, inputs, output_names, system):
  if system == 'torch':
    f = model
  elif system == 'dynamo':
    torch._dynamo.reset()
    f = torch.compile(model)
  elif system == 'torchinductor':
    torch._dynamo.reset()
    f = torch.compile(model, backend='inductor')
  elif system == 'tensorrt':
    from asuka_exp.trtllm_utils import trt_build_engine_from_onnx, trt_build_independent_runtime
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
    )
    engine = trt_build_engine_from_onnx(onnx_model)
    f = trt_build_independent_runtime(engine)

  elif system == 'flashtensor':
    if model.__class__.__name__ == 'H2O':
      from kernel.h2o import H2O
      f = H2O
  elif system == 'ours':
    # This will be handled in test.py with special logic
    f = None
  else:
    raise NotImplementedError(f"system {system} not implemented")

  return f
