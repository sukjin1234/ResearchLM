import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # rank 0 -> 1
    # rank 1 -> 2
    value = torch.tensor(
        [dist.get_rank() + 1],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[before] rank={dist.get_rank()} "
        f"device={device} "
        f"value={value.item()}"
    )

    # 모든 GPU의 value를 더함.
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    print(
        f"[after ] rank={dist.get_rank()} "
        f"device={device} "
        f"value={value.item()}"
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
