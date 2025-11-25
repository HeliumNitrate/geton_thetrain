import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TrainMultiClass():
    
    def __init__(
        self, model, num_epochs, train_dl, valid_dl, loss_fn,
        optimizer, batch_size, scheduler = None
    ):        
        self.model = model
        self.num_epochs = num_epochs
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.loss_hist_tr = [0] * num_epochs
        self.acc_hist_tr = [0] * num_epochs
        self.loss_hist_val = [0] * num_epochs
        self.acc_hist_val = [0] * num_epochs
        
    def train_in_loop(self):
        
        for epoch in range(self.num_epochs):
            batch_index_tr = 0
            batch_index_val = 0
            
            self.model.train()
            for x_batch, y_batch in self.train_dl:
                
                batch_index_tr = self.batch_counter(
                    epoch, batch_index_tr, self.train_dl
                )                
                self.acc_hist_tr, self.loss_hist_tr = self.train(
                    epoch, x_batch, y_batch
                )
            self.loss_hist_tr[epoch] /= len(self.train_dl.dataset)
            self.acc_hist_tr[epoch] /= len(self.train_dl.dataset)
            
            self.model.eval()
            with torch.no_grad():
                for x_batch, y_batch in self.valid_dl:
                    
                    batch_index_val = self.batch_counter(
                        epoch, batch_index_val, self.valid_dl
                    )
                    self.acc_hist_val, self.loss_hist_val = self.validation(
                        epoch, x_batch, y_batch
                    )
            
            self.loss_hist_val[epoch] /= len(self.valid_dl.dataset)
            self.acc_hist_val[epoch] /= len(self.valid_dl.dataset)
            
            self.printing_results(epoch)
            
        return (
            self.loss_hist_tr, self.loss_hist_val, 
            self.acc_hist_tr, self.acc_hist_val
        )
        
    def train(self, epoch, x_batch, y_batch):
        
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(x_batch)
        loss = self.loss_fn(pred, y_batch)
        loss.backward()
        self.optimizer.step()
        self.loss_hist_tr[epoch] += loss.item() * y_batch.size(0)
        
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        self.acc_hist_tr[epoch] += is_correct.sum()
        
        return self.acc_hist_tr, self.loss_hist_tr
        
    
    def validation(self, epoch, x_batch, y_batch):
        
        pred = self.model(x_batch)
        loss = self.loss_fn(pred, y_batch)
        self.loss_hist_val[epoch] += loss.item() * y_batch.size(0)
        
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        self.acc_hist_val[epoch] += is_correct.sum()
        
        return self.acc_hist_val, self.loss_hist_val
        
    def printing_results(self, epoch):
        
        if self.scheduler is not None:
            self.scheduler.step(self.loss_hist_val[epoch])
            current_lr = self.optimizer.param_groups[0]['lr']
        else:
            current_lr = self.optimizer.param_groups[0]['lr']
            
        print(
            f"Epoch {epoch + 1}: ",
            f" tr_accuracy: {self.acc_hist_tr[epoch]:.4f}",
            f" val_accuracy: {self.acc_hist_val[epoch]:.4f}",
            f" tr_loss: {self.loss_hist_tr[epoch]:.4f}",
            f" val_loss: {self.loss_hist_val[epoch]:.4f}",
            f" LR = {current_lr:.2e}"
        )
    
    def batch_counter(self, epoch, batch_index, dl):
        
        batch_index += 1
        num_of_batches = int(np.ceil(len(dl.dataset) / self.batch_size))
        
        if np.remainder(batch_index, 10) == 0:
            print(
                f"Epoch {epoch + 1} / {self.num_epochs}",
                f" Batch {batch_index} / {num_of_batches}"
            )
