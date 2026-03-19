import os
import torch
import torch.distributed as dist


def setup_distributed():
    """
    Initializes the process group if torchrun environment variables exist.
    Retorna:
        ddp_enabled (bool): Se o DDP foi ativado.
        local_rank (int): The GPU ID in this machine (0, 1, ...).
        device (torch.device): O dispositivo configurado.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Estamos rodando via torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)

        # Sincroniza todos os processos
        dist.barrier()

        print(f"Processo DDP iniciado: Rank {rank}/{world_size}, GPU Local {local_rank}")
        return True, local_rank, torch.device(f"cuda:{local_rank}")

    else:
        # Standard execution (python train_flow.py)
        print("Single-GPU execution (DDP disabled).")
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Returns True if it is the master process (Rank 0) or if DDP is not used."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
