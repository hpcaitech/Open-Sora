import time
import thop
from copy import deepcopy

import colossalai
import torch
import torch_musa
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn

from opensora.acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
from opensora.models.text_encoder.t5 import T5Embedder
from transformers.models.t5.modeling_t5 import T5Attention,  T5LayerSelfAttention, T5LayerNorm, T5LayerFF, T5DenseGatedActDense, T5Config, T5Block
from transformers.activations import GELUActivation, NewGELUActivation

# It's a issue founded in T5 encode with fp16; 
# In T5LayerNorm, for some special val in hidden state, its variance will out the range of fp16, which lead to the inverse of var equal to nan; 

# Step 1: base op
# 1.Linear(in_features=4096, out_features=4096, bias=False)
# 2.Linear(in_features=4096, out_features=10240, bias=False)
# 3.(dropout): Dropout(p=0.1, inplace=False)
# 4.(layer_norm): T5LayerNorm()
# 5.(act): NewGELUActivation()

# Step 2: op composed by base op
# 6.T5LayerSelfAttention
# 6.1.T5Attention
# 7.T5DenseGatedActDense
# 8.T5LayerFF

# Step 3: T5Block
# 9.T5Block


# 1.Linear(in_features=4096, out_features=4096, bias=False)
def test_linear1():
    linear1 = torch.nn.Linear(in_features=4096, out_features=4096, dtype=torch.bfloat16, bias=False).musa()
    input = torch.randn(1,120,4096, dtype=torch.bfloat16).musa()
    input.requires_grad=True
    out = linear1(input)
    
    print(out)
    out.mean().backward()


# 2.Linear(in_features=4096, out_features=10240, bias=False)
def test_linear2():
    linear1 = torch.nn.Linear(in_features=4096, out_features=10240, dtype=torch.bfloat16, bias=False).musa()
    input = torch.randn(1,120,4096, dtype=torch.bfloat16).musa()
    input.requires_grad=True
    out = linear1(input)
    
    print(out)
    out.mean().backward()

# 3.(dropout): Dropout(p=0.1, inplace=False)
def test_dropout():
    dropout_layer = torch.nn.Dropout(p=0.1, inplace=False).musa()
    input = torch.randn(1,120,4096, dtype=torch.bfloat16).musa()
    input.requires_grad=True
    out = dropout_layer(input)
    
    print(out)
    out.mean().backward()
    
# 4.(layer_norm): T5LayerNorm()
def test_t5layernorm():
    model = T5LayerNorm(hidden_size=4096).musa()
    input = torch.randn(1,120,4096, dtype=torch.bfloat16).musa()
    input.requires_grad=True
    out = model(input)
    
    print(out)
    out.mean().backward()



# 5.(act): NewGELUActivation()
def test_GELUActivation_use_gelu_python():
    dtype=torch.bfloat16
    model = GELUActivation(use_gelu_python=True).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(f"use_gelu_python out {out}")  # (hidden_states,) + attention_output[1:]
    out.mean().backward()
    
    
# 5.(act): NewGELUActivation()
# torch.erf error
def test_GELUActivation_torch_gelu():
    dtype=torch.bfloat16
    model = GELUActivation(use_gelu_python=False).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(f"no_gelu_python out {out}")  # (hidden_states,) + attention_output[1:]
    out.mean().backward()

# 5 NewGELUActivation
def test_NewGELUActivation():
    dtype=torch.bfloat16
    model = NewGELUActivation().to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(f"NewGELUActivation out {out}")  # (hidden_states,) + attention_output[1:]
    out.mean().backward()

# 6.T5LayerSelfAttention
# none in output 3
def test_T5LayerSelfAttention():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,dtype=dtype)
    model = T5LayerSelfAttention(config).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out[0])  # (hidden_states,) + attention_output[1:]
    out[0].mean().backward()

# 6.1 test T5Attention has_relative_attention_bias
def test_T5Attention_has_relative_attention_bias():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,dtype=dtype)
    model = T5Attention(config=config, has_relative_attention_bias=True).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out[0].shape, out[2].shape)  # (hidden_states,) + attention_output[1:]
    print(out[0], out[2])
    out[0].mean().backward()
    
    
# 6.1 test T5Attention no_relative_attention_bias
# seem nan happen here
def test_T5Attention_no_relative_attention_bias():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,dtype=dtype)
    model = T5Attention(config=config, has_relative_attention_bias=False).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out[0].shape, out[2].shape)  # (hidden_states,) + attention_output[1:]
    print(out[0], out[2])
    out[0].mean().backward()


# 7.T5DenseGatedActDense
# RuntimeError: "threshold_backward" not implemented for 'BFloat16'
def test_T5DenseGatedActDense():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,dtype=dtype)
    model = T5DenseGatedActDense(config).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out)
    out.mean().backward()

# 8.1 T5LayerFF is_gated_act
# RuntimeError: "threshold_backward" not implemented for 'BFloat16'
def test_T5LayerFF_is_gated_act():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,is_gated_act=True, dtype=dtype)
    model = T5LayerFF(config).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out)
    out.mean().backward()

# 8.2 T5LayerFF no_gated_act
# RuntimeError: "threshold_backward" not implemented for 'BFloat16'
def test_T5LayerFF_no_gated_act():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,is_gated_act=False,dtype=dtype)
    model = T5LayerFF(config).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out)
    out.mean().backward()


# 9.T5block
def test_T5Block():
    dtype=torch.bfloat16
    config = T5Config(d_model=4096,dtype=dtype)
    model = T5Block(config, has_relative_attention_bias=True).to(dtype=dtype, device='musa')
    input = torch.randn(1,120,4096).to(dtype=dtype, device='musa')
    input.requires_grad=True
    out = model(input)
    
    print(out[0].shape, out[1].shape)  # (hidden_states,) + attention_output[1:]
    print(out[0], out[1])
    out[0].mean().backward()



def test_t5_encoder():
    # spawn(test_linear1)
    # spawn(run_t5_flops)
    # spawn(run_linear_flops)
    
    test_linear1()
    test_linear2()
    test_dropout()
    test_t5layernorm()
    # test_GELUActivation_use_gelu_python()
    test_GELUActivation_torch_gelu()
    test_NewGELUActivation()
    test_T5LayerSelfAttention()
    test_T5Attention_has_relative_attention_bias()
    test_T5Attention_no_relative_attention_bias()
    test_T5DenseGatedActDense()
    test_T5LayerFF_is_gated_act()
    test_T5LayerFF_no_gated_act()
    test_T5Block()

if __name__ == "__main__":
    test_t5_encoder()
