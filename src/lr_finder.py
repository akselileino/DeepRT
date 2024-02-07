import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

class LRFinder:
    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def range_test(self, train_loader, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):
        lrs = []
        losses = []
        best_loss = float('inf')
        
        # Set model to training mode
        self.model.train()
        
        # Move model to the correct device
        self.model.to(self.device)
        
        # Prepare the data loader
        if not isinstance(train_loader, DataLoader):
            raise ValueError("`train_loader` must be a DataLoader")
        
        # Start from a very small learning rate
        lr = 1e-7
        self.optimizer.param_groups[0]['lr'] = lr
        
        iter_count = 0
        for inputs, labels in train_loader:
            # Break if the number of iterations exceeds the specified number
            if iter_count > num_iter:
                break
            
            # Move data to the correct device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and update
            loss.backward()
            self.optimizer.step()
            
            # Update the learning rate
            lr *= (end_lr / lr) ** (1 / num_iter)
            self.optimizer.param_groups[0]['lr'] = lr
            
            # Record the best loss
            if loss < best_loss:
                best_loss = loss
            
            # Record the learning rate and loss
            lrs.append(lr)
            losses.append(loss.item())
            
            # Check for the "divergence" of the loss
            if loss.item() > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
            
            iter_count += 1
            
        self.plot(lrs, losses, smooth_f)
        
    def plot(self, lrs, losses, smooth_f=0.05):
        # Smoothing the loss
        smoothed_losses = [np.mean(losses[max(0, i-int(smooth_f*len(losses))):i+1]) for i in range(len(losses))]
        plt.plot(lrs[10:-5], smoothed_losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()
