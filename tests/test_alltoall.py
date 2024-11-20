import torch
import torch_musa
import torch.distributed as dist


def cleanup():
    dist.destroy_process_group()

def run_alltoall(rank, world_size):
    dist.init_process_group(backend="mccl")
    
    # 每个进程生成一个唯一的张量
    local_data = [torch.rand(10).to('musa') for _ in range(world_size)]
    
    # 存储接收到的数据
    received_data = [torch.empty(10).to('musa') for _ in range(world_size)]
    
    # 使用 alltoall 函数进行数据交换
    dist.all_to_all(received_data, local_data)
    
    # 打印结果
    print(f"Rank {rank} received data: {received_data}")
    
    cleanup()

def run_alltoall_single(rank, world_size):
    dist.init_process_group(backend="mccl")
    
    # 每个进程生成一个唯一的张量
    local_data = torch.arange(4) + rank * 4
    local_data = local_data.to(device="musa", dtype=torch.bfloat16)
    
    # 存储接收到的数据
    received_data = torch.empty([4])
    received_data = received_data.to(device="musa", dtype=torch.bfloat16)
    
    # 使用 alltoall 函数进行数据交换
    dist.all_to_all_single(received_data, local_data)
    
    # 打印结果
    print(f"Rank {rank} received data: {received_data}")
    
    cleanup()

def main():
    world_size = 4  # 假设有4个进程
    for rank in range(world_size):
        # run_alltoall(rank, world_size)
        run_alltoall_single(rank, world_size)

if __name__ == "__main__":
    main()