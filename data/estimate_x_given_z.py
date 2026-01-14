import torch
import torch.nn as nn
import torch.optim as optim

class mu_X_Given_Z_Estimator(nn.Module):
    def __init__(self, input_dim=19, hidden_size=128, output_size=1, 
                 layer_norm=True, drop_out=True, drop_out_p=0.3):
        super().__init__()
        # Support both single int and list of ints for hidden_size
        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = list(hidden_size)
        
        # Build hidden layers
        layers = []
        prev_size = input_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            if layer_norm:
                layers.append(nn.LayerNorm(h_size))
            layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(drop_out_p))
            prev_size = h_size
        
        self.shared = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, output_size)

    def forward(self, z):
        h = self.shared(z)
        mu = self.output(h)
        return mu


def train_estimator(model, Z_train, gt_mu, X_train,
                    epochs=500, lr=0.001,
                    optimizer=None):
    """
    Train an estimator for E[X|Z].
    
    Args:
    - model: mu_X_Given_Z_Estimator model to train
    - Z_train: Training conditioning variables
    - gt_mu: Ground truth mean (for debugging loss, can be None for real data)
    - X_train: Training target variable
    - epochs: Maximum number of training epochs
    - lr: Learning rate
    - optimizer: Existing optimizer (optional, for online mode to maintain momentum)
    
    Returns:
    - optimizer: Optimizer (for continuing training in online mode)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Handle case where gt_mu is None (real data without ground truth)
    has_gt_mu = gt_mu is not None
    if has_gt_mu:
        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, gt_mu, X_train), 
            batch_size=128, shuffle=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_train, X_train), 
            batch_size=128, shuffle=True)
    
    # Create or reuse optimizer (reuse maintains momentum for warm-starting)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        gt_loss_sum = 0.0
        
        if has_gt_mu:
            for Z_batch, gt_mu_batch, X_batch in train_dataloader:
                Z_batch = Z_batch.to(device)
                gt_mu_batch = gt_mu_batch.to(device)
                X_batch = X_batch.to(device)
                
                optimizer.zero_grad()
                mu = model(Z_batch)
                loss = criterion(mu, X_batch)
                gt_loss = criterion(mu, gt_mu_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(X_batch)
                gt_loss_sum += gt_loss.item() * len(X_batch)
        else:
            for Z_batch, X_batch in train_dataloader:
                Z_batch = Z_batch.to(device)
                X_batch = X_batch.to(device)
                
                optimizer.zero_grad()
                mu = model(Z_batch)
                loss = criterion(mu, X_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_dataloader.dataset)
        gt_loss_avg = gt_loss_sum / len(train_dataloader.dataset) if has_gt_mu else None
        
        # Print training progress
        if has_gt_mu:
            print(f"Epoch {epoch+1}, Train: {train_loss:.4f}, GT: {gt_loss_avg:.4f}", end='\r')
        else:
            print(f"Epoch {epoch+1}, Train: {train_loss:.4f}", end='\r')
    
    print()  # New line after training
    model.eval()
    return optimizer
