import torch
import torch_musa
from torch.testing import assert_close

musa_infer = torch.load('/home/dist/hpcai/duanjunwen/Open-Sora/samples/samples-v1-1/infer_output_musa_fp32_1.pt', map_location=torch.device('cpu'))
nv_infer= torch.load('/home/dist/hpcai/duanjunwen/Open-Sora/samples/samples-v1-1/infer_output_nv_fp32_2.pt', map_location=torch.device('cpu'))
musa_infer = musa_infer.to('cpu')
nv_infer = nv_infer.to('cpu')
assert_close(musa_infer, nv_infer, rtol=1.6e-2, atol=1e-5)