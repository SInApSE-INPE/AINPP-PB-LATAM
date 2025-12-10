import torch
import os
import mlflow
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, rank=0, local_rank=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        
        if config.training.distributed:
            self.device = torch.device(f'cuda:{local_rank}')
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[local_rank])
        else:
            self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        
        self.criterion = nn.MSELoss() # Assuming regression for nowcasting
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        
        self.epochs = config.training.epochs

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log step-level metrics
            if self.rank == 0:
                mlflow.log_metric("train_step_loss", loss.item(), step=epoch * len(self.train_loader) + batch_idx)

        epoch_loss = running_loss / len(self.train_loader)
        if self.rank == 0:
            mlflow.log_metric("train_epoch_loss", epoch_loss, step=epoch)
        return epoch_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(self.val_loader)
        if self.rank == 0:
            mlflow.log_metric("val_epoch_loss", epoch_loss, step=epoch)
        return epoch_loss

    def fit(self, train_sampler=None):
        if self.rank == 0:
            mlflow.set_experiment(self.config.training.experiment_name)
            mlflow.start_run()
            # Log params
            mlflow.log_params(self.config.training)
            mlflow.log_params(self.config.model)
            mlflow.log_params(self.config.dataset)
            
        for epoch in range(self.epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)
                
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            if self.rank == 0:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save checkpoint
                os.makedirs("outputs", exist_ok=True)
                torch.save(self.model.state_dict(), f"outputs/checkpoint_epoch_{epoch+1}.pth")
        
        if self.rank == 0:
            mlflow.end_run()
