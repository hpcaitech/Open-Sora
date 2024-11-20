import torch
import torch_musa
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from collections import OrderedDict
from torch.testing import assert_close

GLOBAL_STEP = 1

def assert_param(musa_model:OrderedDict, torch_model:OrderedDict):
    for (musa_key, musa_val), (torch_key, torch_val) in zip(musa_model.items(), torch_model.items()):
        if musa_key == torch_key:
            assert_close(musa_val, torch_val)
        

def assert_optim_state(musa_optim_state:dict, torch_optim_state:dict, step: int):
    
    # check state 
    for (musa_key, musa_state), (torch_key, torch_state) in zip(musa_optim_state['state'].items(), torch_optim_state['state'].items()):
        # musa_state:dict; key --> step, exp_avg, exp_avg_sq
        for (musa_state_name, musa_state_val), (torch_state_name, torch_state_val) in zip(musa_state.items(), torch_state.items()):
            print(f"Check {musa_key} {musa_state_name} ")
            if musa_key == torch_key and musa_state_name == torch_state_name:
                assert_close(musa_state_val, torch_state_val)
    print(f"optimizer state step{step} stage assert close pass;")
    
    # check param group
    # musa_optim_state['param_groups']--> type: list; save each param_groups
    for musa_param_groups, torch_param_groups in zip(musa_optim_state['param_groups'], torch_optim_state['param_groups']):
        for (musa_key, musa_param), (torch_key, torch_param) in zip(musa_param_groups.items(), torch_param_groups.items()):
            if musa_key == 'params' and musa_key == torch_key:
                assert_close(musa_param, torch_param)
    print(f"optimizer param_group step{step} stage assert close pass;")


def assert_tensor(musa_tensor:torch.Tensor, torch_tensor:torch.Tensor):
    assert_close(musa_tensor, torch_tensor)
    
###################
# assert model in training
###################
def assert_model(model_name:str):
    assert model_name in ['vae', 't5', 'stdit'], 'Model not in support list.'
    # 1.assert vae, stdit param in init stage; 
    if model_name in ['vae', 'stdit']:
        model_param_init_musa = torch.load(f"./dataset/assert_closed/musa_tensor/{model_name}_param_init.txt", map_location=torch.device('musa'))
        model_param_init_torch = torch.load(f"./dataset/assert_closed/torch_tensor/{model_name}_param_init.txt", map_location=torch.device('musa'))
        assert_param(model_param_init_musa, model_param_init_torch)
        print(f"{model_name} init stage assert close pass;")
    
    # 2.assert vae, t5 input&output in step, stdit param, optim state
    global_step = GLOBAL_STEP
    for step in range(global_step):
        if model_name in ['vae', 't5']:
            # 2.1 assert input
            model_input_musa = torch.load(f"./dataset/assert_closed/musa_tensor/step{step}_{model_name}_input.txt", map_location=torch.device('musa'))
            model_input_torch = torch.load(f"./dataset/assert_closed/torch_tensor/step{step}_{model_name}_input.txt", map_location=torch.device('musa'))
            if model_name == 't5':
                assert model_input_musa == model_input_torch
            else:
                assert_tensor(model_input_musa, model_input_torch)
            print(f"{model_name} input step{step} assert close pass;")
            
            
            # 2.2 assert output
            model_output_musa = torch.load(f"./dataset/assert_closed/musa_tensor/step{step}_{model_name}_output.txt", map_location=torch.device('musa'))
            model_output_torch = torch.load(f"./dataset/assert_closed/torch_tensor/step{step}_{model_name}_output.txt", map_location=torch.device('musa'))
            
            if model_name == 't5':
                # y, mask
                for (musa_key, musa_param), (torch_key, torch_param) in zip(model_output_musa.items(), model_output_torch.items()):
                    if musa_key == torch_key:
                        # TODO: torch_param is fp16 now, should be bf16;
                        torch_param = torch_param.to(dtype=torch.bfloat16)
                        assert_tensor(musa_param, torch_param)
            else:
                assert_tensor(model_output_musa, model_output_torch)
            print(f"{model_name} output step{step} assert close pass;")
        else: 
            model_param_step_musa = torch.load(f"./dataset/assert_closed/musa_tensor/step{step}_{model_name}_param_step.txt", map_location=torch.device('musa'))
            model_param_step_torch = torch.load(f"./dataset/assert_closed/torch_tensor/step{step}_{model_name}_param_step.txt", map_location=torch.device('musa'))
            assert_param(model_param_step_musa, model_param_step_torch)
            
###################
# assert optim in training
###################
def assert_optimizer():
    # 1.assert optim state init
    optim_state_init_musa = torch.load(f"./dataset/assert_closed/musa_tensor/optim_state_init.txt", map_location=torch.device('musa'))
    optim_state_init_torch = torch.load(f"./dataset/assert_closed/torch_tensor/optim_state_init.txt", map_location=torch.device('musa'))
    assert_optim_state(optim_state_init_musa, optim_state_init_torch, 0)
    print(f"optimizer state init stage assert close pass;")
    # 2.assert optim state in step
    global_step = GLOBAL_STEP
    for step in range(global_step):
        optim_state_step_musa = torch.load(f"./dataset/assert_closed/musa_tensor/step{step}_optim_state_step.txt", map_location=torch.device('musa'))
        optim_state_step_torch = torch.load(f"./dataset/assert_closed/torch_tensor/step{step}_optim_state_step.txt", map_location=torch.device('musa'))
        assert_optim_state(optim_state_step_musa, optim_state_step_torch, step)




if __name__ == "__main__":
    # assert_model('vae')
    # assert_model('stdit')
    # assert_model('t5')
    assert_optimizer()