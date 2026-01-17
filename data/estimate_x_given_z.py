import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Estimator type constants
ESTIMATOR_MLP = "mlp"
ESTIMATOR_GMMN = "gmmn"


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


class GMMN_Estimator(nn.Module):
    """
    Generative Moment Matching Network for estimating P(X|Z).
    
    GMMN (Dziugaite et al., 2015; Li et al., 2015) learns a generator G that maps
    noise η ~ N(0, I_m) and conditioning variable z to samples from P(X|Z=z).
    
    The generator is trained to minimize Maximum Mean Discrepancy (MMD) between
    generated samples and true samples from P(X|Z).
    
    For sampling: given z, sample η ~ N(0, I_m) and output G(η, z).
    """
    def __init__(self, input_dim=19, hidden_size=128, output_size=1, 
                 drop_out=False, drop_out_p=0.3,
                 noise_dim=16, layer_norm=True):
        super().__init__()
        self.input_dim = input_dim  # z_dim (conditioning variable)
        self.output_size = output_size  # x_dim (target variable)
        self.noise_dim = noise_dim  # dimension of noise η
        
        # Support both single int and list of ints for hidden_size
        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = list(hidden_size)
        
        # Build generator: (η, Z) -> X
        layers = []
        prev_size = noise_dim + input_dim  # η + Z
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            if layer_norm:
                layers.append(nn.LayerNorm(h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.generator = nn.Sequential(*layers)

    def forward(self, eta, z):
        """
        Forward pass: generate X from noise η and conditioning variable z.
        
        Args:
            eta: Noise samples (batch, noise_dim)
            z: Conditioning variables (batch, input_dim)
            
        Returns:
            x: Generated samples (batch, output_size)
        """
        return self.generator(torch.cat([eta, z], dim=-1))

    def sample(self, z, n_samples=1):
        """
        Sample from P(X|Z) by sampling noise and passing through generator.
        
        Args:
            z: Conditioning variable (batch, input_dim)
            n_samples: Number of samples per z (default 1)
            
        Returns:
            samples: Sampled X values (batch, output_size) if n_samples=1
                     else (batch, n_samples, output_size)
        """
        batch_size = z.shape[0]
        device = z.device
        
        if n_samples == 1:
            eta = torch.randn(batch_size, self.noise_dim, device=device)
            return self.forward(eta, z)
        else:
            # Use vectorized sample_multiple for efficiency
            return self.sample_multiple(z, n_samples)

    def sample_multiple(self, z, M):
        """
        Generates M samples for each z in the batch (vectorized).
        
        Args:
            z: Conditioning variable (batch_size, input_dim)
            M: Number of samples per z
            
        Returns:
            x_gen: Generated samples (batch_size, M, output_size)
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Repeat Z: [z1, z2] -> [z1, z1, ..., z2, z2, ...]
        z_repeated = z.repeat_interleave(M, dim=0)  # (batch_size * M, input_dim)
        
        # Sample noise for all repetitions
        eta = torch.randn(batch_size * M, self.noise_dim, device=device)
        
        # Generate and reshape
        samples = self.forward(eta, z_repeated)  # (batch_size * M, output_size)
        return samples.view(batch_size, M, self.output_size)


def compute_kernel(x, y, kernel_type='rbf', bandwidth=None):
    """
    Compute kernel matrix between x and y.
    
    Args:
        x: First set of samples (n, d)
        y: Second set of samples (m, d)
        kernel_type: Type of kernel ('rbf' or 'imq', 'linear')
        bandwidth: Kernel bandwidth (if None, use median heuristic)
        
    Returns:
        K: Kernel matrix (n, m)
    """
    # Compute pairwise squared distances
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (n, 1)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # (m, 1)
    dist_sq = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())  # (n, m)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    
    if bandwidth is None and kernel_type != 'linear':
        # Median heuristic for bandwidth
        with torch.no_grad():
            median_dist = torch.median(dist_sq[dist_sq > 0])
            bandwidth = median_dist.item() if median_dist > 0 else 1.0
    
    if kernel_type == 'rbf':
        # RBF kernel: exp(-||x-y||^2 / (2 * bandwidth))
        K = torch.exp(-dist_sq / (2 * bandwidth))
    elif kernel_type == 'imq':
        # Inverse multiquadric kernel: 1 / sqrt(1 + ||x-y||^2 / bandwidth)
        K = 1.0 / torch.sqrt(1.0 + dist_sq / bandwidth)
    elif kernel_type == 'linear':
        # Linear kernel: x^T y
        K = torch.mm(x, y.t())
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return K


def mmd_loss_conditional(x_real, x_gen, z, kernel_type='rbf', bandwidth_x=None, bandwidth_z=None, M=1):
    """
    Compute Conditional MMD loss with M generated samples per conditioning variable.
    
    Args:
        x_real: Real samples (n, d_x)
        x_gen: Generated samples (n, d_x) if M=1, or (n, M, d_x) if M>1
        z: Conditioning variables (n, d_z)
        kernel_type: Type of kernel for X space
        bandwidth_x: Bandwidth for the X kernels (target)
        bandwidth_z: Bandwidth for the Z kernel (weighting)
        M: Number of generated samples per conditioning variable (default 1)
        
    Returns:
        loss: Scalar CMMD loss
    """
    n = x_real.shape[0]
    device = x_real.device
    
    # 1. Compute the Weighting Matrix (W) in Z-space
    # This determines "how much" we care about the pair (i, j)
    K_zz = compute_kernel(z, z, kernel_type='rbf', bandwidth=bandwidth_z)
    
    # 2. Compute K_xx: kernel between real samples
    K_xx = compute_kernel(x_real, x_real, kernel_type, bandwidth_x)
    
    # 3. Handle M generated samples per conditioning variable
    if M == 1 or x_gen.dim() == 2:
        # Original behavior: 1:1 match between x_real and x_gen
        K_yy = compute_kernel(x_gen, x_gen, kernel_type, bandwidth_x)
        K_xy = compute_kernel(x_real, x_gen, kernel_type, bandwidth_x)
    else:
        # M samples per z: x_gen is (n, M, d_x)
        d_x = x_real.shape[-1]
        
        # Flatten x_gen for kernel computation
        x_gen_flat = x_gen.view(n * M, d_x)  # (n*M, d_x)
        
        # Compute K_yy: kernel between all generated samples, then average
        K_yy_full = compute_kernel(x_gen_flat, x_gen_flat, kernel_type, bandwidth_x)  # (n*M, n*M)
        K_yy_full = K_yy_full.view(n, M, n, M)
        K_yy = K_yy_full.mean(dim=(1, 3))  # (n, n) - average over M×M pairs
        
        # Compute K_xy: kernel between real and generated, then average over M
        K_xy_full = compute_kernel(x_real, x_gen_flat, kernel_type, bandwidth_x)  # (n, n*M)
        K_xy_full = K_xy_full.view(n, n, M)
        K_xy = K_xy_full.mean(dim=2)  # (n, n) - average over M samples
    
    # 4. Compute the Discrepancy Matrix (U) in X-space
    # Following the logic: U = K_xx - K_xy - K_yx + K_yy
    U_mx = K_xx - K_xy - K_xy.t() + K_yy
    
    # 5. Apply Weighting and Remove Self-Similarity (Diagonal)
    # We only care about how x_real[i] relates to x_gen[j] 
    # when z[i] is close to z[j].
    weighted_discrepancy = U_mx * K_zz
    
    # Mask out diagonal to follow unbiased-style sum
    mask = 1.0 - torch.eye(n, device=device)
    loss = (weighted_discrepancy * mask).sum() / (n * (n - 1))
    
    return loss

def train_gmmn_estimator(model, Z_train, X_train,
                         epochs=500, lr=0.001, M_train=5,
                         kernel_type='rbf', optimizer=None, grad_clip=0.5):
    """
    Train a GMMN estimator for P(X|Z) using CMMD loss with M samples per Z.
    
    Args:
        model: GMMN_Estimator model to train
        Z_train: Training conditioning variables (n, z_dim)
        X_train: Training target variable (n, x_dim)
        epochs: Number of training epochs
        lr: Learning rate
        M_train: Number of generated samples per Z for training (default 5)
        kernel_type: Type of kernel for MMD ('rbf' or 'imq')
        optimizer: Existing optimizer (optional, for online mode)
        grad_clip: Max gradient norm for clipping (default 0.5)
    
    Returns:
        optimizer: Optimizer (for continuing training in online mode)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Z_train, X_train), 
        batch_size=20, shuffle=True)
    
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for Z_batch, X_batch in train_dataloader:
            Z_batch = Z_batch.to(device)
            X_batch = X_batch.to(device)
            batch_size = Z_batch.shape[0]
            
            optimizer.zero_grad()
            
            # Generate M samples per Z
            X_fake = model.sample_multiple(Z_batch, M_train)  # (batch_size, M, output_size)
            
            # Flatten X dimensions if needed
            X_real = X_batch.view(batch_size, -1)
            Z_real = Z_batch.view(batch_size, -1)
            
            # Compute CMMD loss with M samples
            loss = mmd_loss_conditional(X_real, X_fake, Z_real, kernel_type=kernel_type, 
                                        M=M_train, bandwidth_x=None, bandwidth_z=None)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item() * batch_size
        
        train_loss /= len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}, GMMN CMMD Loss: {train_loss:.6f}", end='\r')
    
    print()  # New line after training
    model.eval()
    return optimizer


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
