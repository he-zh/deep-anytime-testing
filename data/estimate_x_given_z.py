import torch
import torch.nn as nn
import torch.optim as optim

class PX_Given_Z_Estimator(nn.Module):
    def __init__(self, input_dim=19):
        super().__init__()
        # Shared layers to learn Z representations
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
        )
        # Head to predict the mean
        self.output = nn.Linear(128, 1)

    def forward(self, z):
        h = self.shared(z)
        mu = self.output(h)
        return mu

def train_estimator(Z_train, gt_mu, X_train, epochs=500):
    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Z_train, gt_mu, X_train), 
                                                   batch_size=128, shuffle=True)

    model = PX_Given_Z_Estimator(input_dim=Z_train.shape[1])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss: Negative Log Likelihood of Gaussian
    # This is better than MSE because it learns the noise level (sigma)
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for Z_batch, gt_mu_batch, X_batch in train_dataloader:
            Z_batch = Z_batch.to("cuda" if torch.cuda.is_available() else "cpu")
            gt_mu_batch = gt_mu_batch.to("cuda" if torch.cuda.is_available() else "cpu")
            X_batch = X_batch.to("cuda" if torch.cuda.is_available() else "cpu")
            
            optimizer.zero_grad()
            mu = model(Z_batch)
            
            # # Gaussian NLL loss
            # precision = torch.exp(-logvar)
            # loss = torch.mean(0.5 * precision * (X_batch - mu)**2 + 0.5 * logvar)
            loss = criterion(mu, X_batch)

            gt_loss = criterion(mu, gt_mu_batch)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, GT Loss: {gt_loss.item():.4f}", end='\r')

    model.eval()
    return model
