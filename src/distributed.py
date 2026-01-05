import os
import torch
import torch.distributed as dist

def setup_distributed():
    """
    Inicializa o grupo de processos se as variáveis de ambiente do torchrun existirem.
    Retorna:
        ddp_enabled (bool): Se o DDP foi ativado.
        local_rank (int): O ID da GPU nesta máquina (0, 1, ...).
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
        # Execução padrão (python train_flow.py)
        print("Execução Single-GPU (DDP desativado).")
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Retorna True se for o processo mestre (Rank 0) ou se não estiver usando DDP."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0