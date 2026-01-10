import torch
import torch.nn as nn
import torch.optim as optim
from models.earlystopping import EarlyStopper

class PX_Given_Z_Estimator(nn.Module):
    def __init__(self, input_dim=19, hidden_size=128, output_size=1, 
                 layer_norm=True, drop_out=True, drop_out_p=0.3):
        super().__init__()
        # Configurable architecture similar to MLP
        layers = [nn.Linear(input_dim, hidden_size)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(drop_out_p))
        
        self.shared = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        h = self.shared(z)
        mu = self.output(h)
        return mu


def train_estimator(model, Z_train, gt_mu, X_train, Z_val=None, X_val=None, 
                    epochs=500, patience=10, lr=0.001,
                    optimizer=None, early_stopper=None):
    """
    Train an estimator for E[X|Z].
    
    Args:
    - model: PX_Given_Z_Estimator model to train
    - Z_train: Training conditioning variables
    - gt_mu: Ground truth mean (for debugging loss)
    - X_train: Training target variable
    - Z_val: Validation conditioning variables (optional, for early stopping)
    - X_val: Validation target variable (optional, for early stopping)
    - epochs: Maximum number of training epochs
    - patience: Early stopping patience
    - lr: Learning rate
    - optimizer: Existing optimizer (optional, for online mode)
    - early_stopper: Existing EarlyStopper (optional, for online mode)
    
    Returns:
    - optimizer: Optimizer (for continuing training)
    - early_stopper: EarlyStopper (for continuing training)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Z_train, gt_mu, X_train), 
        batch_size=128, shuffle=True)
    
    # Create validation dataloader if validation data is provided
    val_dataloader = None
    if Z_val is not None and X_val is not None:
        val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z_val, X_val), 
            batch_size=128, shuffle=False)
    
    # Create or reuse optimizer
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create or reuse early stopper (reset for new training session)
    if early_stopper is None:
        early_stopper = EarlyStopper(patience=patience, min_delta=0)
    else:
        early_stopper.reset()
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        gt_loss_sum = 0.0
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
        
        train_loss /= len(train_dataloader.dataset)
        gt_loss_avg = gt_loss_sum / len(train_dataloader.dataset)
        
        # Validation and early stopping
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Z_batch, X_batch in val_dataloader:
                    Z_batch = Z_batch.to(device)
                    X_batch = X_batch.to(device)
                    mu = model(Z_batch)
                    val_loss += criterion(mu, X_batch).item() * len(X_batch)
            val_loss /= len(val_dataloader.dataset)
            
            print(f"Epoch {epoch+1}, Train: {train_loss:.4f}, Val: {val_loss:.4f}, GT: {gt_loss_avg:.4f}", end='\r')
            
            # Early stopping check
            if early_stopper.early_stop(val_loss):
                print(f"\nEarly stopping at epoch {epoch+1}, best val loss: {early_stopper.min_validation_loss:.4f}")
                break
        else:
            print(f"Epoch {epoch+1}, Train: {train_loss:.4f}, GT: {gt_loss_avg:.4f}", end='\r')
    
    print()  # New line after training
    model.eval()
    return optimizer, early_stopper
