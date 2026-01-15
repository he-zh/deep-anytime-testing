from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import numpy as np

# Mode constants for X|Z estimation
MODE_MODEL_X = "model_x"           # Use true μ to sample X̃
MODE_PSEUDO_MODEL_X = "pseudo_model_x"  # Pre-train estimator with extra data
MODE_ONLINE = "online"             # Estimate μ online using training data

# Estimator type constants (re-export from estimate_x_given_z for convenience)
from .estimate_x_given_z import ESTIMATOR_MLP, ESTIMATOR_GMMN


class DataGenerator(ABC):
    def __init__(self, type, samples,data_seed):
        assert type in ["type2", "type11", "type12", "type1"]
        assert samples > 0
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
    @abstractmethod
    def generate(self)->Dataset:
        pass

class DatasetOperator(Dataset):
    def __init__(self, tau1, tau2):
        self.tau1 = tau1
        self.tau2 = tau2

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        tau1_z, tau2_z = self.z[idx], self.z[idx].clone()
        if self.tau1 is not None:
            tau1_z = self.tau1(tau1_z)
        if self.tau2 is not None:
            tau2_z = self.tau2(tau2_z)
        return tau1_z, tau2_z

    def regenerate_tilde(self, estimator, z_dim=1, x_dim=1, estimator_type=ESTIMATOR_MLP,
                         use_shrinkage_cov=False, shrinkage_alpha=0.1, cov_cholesky=None):
        """
        Regenerate X_tilde using an updated estimator.
        
        This method should be called after the online estimator is updated
        to refresh all X_tilde values with the improved estimator.
        
        The data structure in self.z is:
        - z[:, :, 0] = Z = [X_target, Z_cov, Y]
        - z[:, :, 1] = Z_tilde = [X_tilde, Z_cov, Y]
        
        Where X_target is at indices 0:x_dim, Z_cov is at indices x_dim:x_dim+z_dim, Y is at x_dim+z_dim:
        
        Args:
        - estimator: Updated estimator for E[X|Z] or P(X|Z)
        - z_dim: Dimension of conditioning variable Z_cov (default 1)
        - x_dim: Dimension of target variable X (default 1)
        - estimator_type: Type of estimator ('mlp', 'gmmn')
        - use_shrinkage_cov: If True, use shrinkage covariance for MLP sampling
        - shrinkage_alpha: Shrinkage strength (only used when use_shrinkage_cov=True and cov_cholesky is None)
        - cov_cholesky: Pre-computed Cholesky factor of global covariance (d, d). If provided, uses this instead of batch covariance.
        """
        if estimator is None:
            return
            
        # Extract from original Z (z[:, :, 0])
        Z_original = self.z[:, :, 0]
        X_target = Z_original[:, :x_dim]              # (n, x_dim)
        Z_cov = Z_original[:, x_dim:x_dim+z_dim]      # (n, z_dim)
        Y = Z_original[:, x_dim+z_dim:]               # (n, y_dim)
        
        # Regenerate X_tilde with the updated estimator based on type
        X_tilde_new = sample_from_estimator(
            Z_cov, X_target, estimator, estimator_type,
            use_shrinkage_cov=use_shrinkage_cov, shrinkage_alpha=shrinkage_alpha,
            cov_cholesky=cov_cholesky
        ).to(Z_original.device)
        
        # Reconstruct Z_tilde = [X_tilde, Z_cov, Y]
        Z_tilde_new = torch.cat((X_tilde_new, Z_cov, Y), dim=1)
        
        # Update self.z with new Z_tilde
        self.z = torch.stack([Z_original, Z_tilde_new], dim=2)


class MergedDataset(DatasetOperator):
    """
    A dataset that merges multiple DatasetOperator datasets into one.
    Concatenates the underlying z tensors and ground_truth_mu (if present).
    """
    
    def __init__(self, datasets):
        """
        Merge multiple datasets into one.
        
        Args:
        - datasets: List of DatasetOperator instances to merge
        """
        if not datasets:
            raise ValueError("Cannot create MergedDataset from empty list")
        
        # Get tau1, tau2 from first dataset
        first = datasets[0]
        super().__init__(first.tau1, first.tau2)
        
        # Collect all z tensors and ground_truth_mu
        all_z = []
        all_mu = []
        for ds in datasets:
            if hasattr(ds, 'z'):
                all_z.append(ds.z)
            if hasattr(ds, 'ground_truth_mu') and ds.ground_truth_mu is not None:
                all_mu.append(ds.ground_truth_mu)
        
        # Concatenate
        self.z = torch.cat(all_z, dim=0)
        self.ground_truth_mu = torch.cat(all_mu, dim=0) if all_mu else None


class CITDataGeneratorBase(DataGenerator):
    """
    Base class for Conditional Independence Test data generators.
    Handles the three modes: model_x, pseudo_model_x, and online.
    
    Supports two estimator types:
    - 'mlp': Standard MLP that estimates E[X|Z], samples using mean + Gaussian noise
    - 'gmmn': Generative Moment Matching Network that learns P(X|Z) via MMD
    """
    
    def __init__(self, type, samples, data_seed, mode=MODE_MODEL_X, pretrain_samples=5000,
                 estimator_cfg=None):
        """
        Initialize the CIT data generator base.
        
        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate per batch.
        - data_seed (int): Seed for random number generation.
        - mode (str): Mode for X|Z estimation. One of 'model_x', 'pseudo_model_x', 'online'.
        - pretrain_samples (int): Number of samples for pre-training estimator (only used in pseudo_model_x mode).
        - estimator_cfg (dict, optional): Config for estimator model (hidden_size, layer_norm, drop_out, etc.)
        """
        super().__init__(type, samples, data_seed)
        self.type = type
        self.samples = samples
        self.data_seed = data_seed
        self.mode = mode
        self.pretrain_samples = pretrain_samples
        
        # Estimator configuration (with defaults)
        self.estimator_cfg = estimator_cfg or {}
        self.estimator_hidden_size = self.estimator_cfg.get('hidden_size', 128)
        self.estimator_layer_norm = self.estimator_cfg.get('layer_norm', True)
        self.estimator_drop_out = self.estimator_cfg.get('drop_out', True)
        self.estimator_drop_out_p = self.estimator_cfg.get('drop_out_p', 0.3)
        self.estimator_lr = self.estimator_cfg.get('lr', 0.001)
        self.estimator_epochs_pseudo = self.estimator_cfg.get('epochs_pseudo', 1000)  # For pseudo_model_x (trains once with more data)
        self.estimator_epochs_online = self.estimator_cfg.get('epochs_online', 100)   # For online (trains incrementally)
        
        # Estimator type settings (can be overridden by subclasses)
        # Options: 'mlp', 'gmmn'
        self.estimator_type = self.estimator_cfg.get('estimator_type', ESTIMATOR_MLP)   
        self.noise_dim = self.estimator_cfg.get('noise_dim', 16)    # For GMMN
        self.gmmn_kernel = self.estimator_cfg.get('gmmn_kernel', 'rbf')  # Kernel type for GMMN MMD loss
        self.gmmn_M_train = self.estimator_cfg.get('gmmn_M_train', 5)  # Number of samples per Z during GMMN training
        self.gmmn_grad_clip = self.estimator_cfg.get('gmmn_grad_clip', 0.5)  # Gradient clipping for GMMN
        
        # MLP sampling settings
        self.use_shrinkage_cov = self.estimator_cfg.get('use_shrinkage_cov', False)  # Use shrinkage covariance for MLP
        self.shrinkage_alpha = self.estimator_cfg.get('shrinkage_alpha', 0.1)  # Shrinkage strength
        
        # Global covariance estimation (accumulated across all data)
        self._accumulated_residuals = None  # For computing global covariance
        self._global_cov_matrix = None  # Cached covariance matrix
        self._global_cov_cholesky = None  # Cached Cholesky factor for sampling
        
        # Will be set by subclass or during training
        self.estimator = None  # Renamed from mu_X_given_Z_estimator
        self.estimator_optimizer = None
        self.accumulated_Z = None
        self.accumulated_X = None
        self.accumulated_gt_mu = None

    @property
    def z_dim(self):
        """Dimension of conditioning variable Z (c)."""
        return self._z_dim

    @property
    def x_dim(self):
        """
        Return the dimension of the target variable X.
        Default is 1, override in subclasses if different.
        
        Returns:
        - int: Dimension of X (the target variable for X|Z estimation)
        """
        return self._x_dim
        
    def _create_estimator(self):
        """
        Create a new estimator model with the configured architecture.
        
        Returns:
        - model: mu_X_Given_Z_Estimator instance
        """
        from .estimate_x_given_z import mu_X_Given_Z_Estimator
        
        return mu_X_Given_Z_Estimator(
            input_dim=self.z_dim,
            hidden_size=self.estimator_hidden_size,
            output_size=self.x_dim,
            layer_norm=self.estimator_layer_norm,
            drop_out=self.estimator_drop_out,
            drop_out_p=self.estimator_drop_out_p
        )
    
    def _create_estimator_by_type(self, estimator_type=None):
        """
        Factory method to create an estimator based on type.
        
        Args:
        - estimator_type: Type of estimator ('mlp', 'gmmn'). 
                         If None, uses self.estimator_type.
        
        Returns:
        - model: Estimator instance of the specified type
        """
        if estimator_type is None:
            estimator_type = self.estimator_type
            
        if estimator_type == ESTIMATOR_MLP:
            from .estimate_x_given_z import mu_X_Given_Z_Estimator
            return mu_X_Given_Z_Estimator(
                input_dim=self.z_dim,
                hidden_size=self.estimator_hidden_size,
                output_size=self.x_dim,
                layer_norm=self.estimator_layer_norm,
                drop_out=self.estimator_drop_out,
                drop_out_p=self.estimator_drop_out_p
            )
        elif estimator_type == ESTIMATOR_GMMN:
            from .estimate_x_given_z import GMMN_Estimator
            return GMMN_Estimator(
                input_dim=self.z_dim,
                hidden_size=self.estimator_hidden_size,
                output_size=self.x_dim,
                noise_dim=self.noise_dim,
                layer_norm=self.estimator_layer_norm,
                drop_out=self.estimator_drop_out,
                drop_out_p=self.estimator_drop_out_p
            )
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}. Choose from 'mlp', 'gmmn'.")
    
    def _train_estimator_by_type(self, model, Z_train, X_train, gt_mu=None, epochs=None, optimizer=None):
        """
        Train an estimator based on its type.
        
        Args:
        - model: Estimator model to train
        - Z_train: Conditioning variables
        - X_train: Target variables
        - gt_mu: Ground truth mean (only used for MLP debugging)
        - epochs: Number of epochs (uses config default if None)
        - optimizer: Existing optimizer for warm start
        
        Returns:
        - optimizer: Trained optimizer
        """
        if epochs is None:
            epochs = self.estimator_epochs_pseudo
            
        if self.estimator_type == ESTIMATOR_MLP:
            from .estimate_x_given_z import train_estimator
            return train_estimator(
                model, Z_train, gt_mu, X_train,
                epochs=epochs, lr=self.estimator_lr, optimizer=optimizer
            )
        elif self.estimator_type == ESTIMATOR_GMMN:
            from .estimate_x_given_z import train_gmmn_estimator
            return train_gmmn_estimator(
                model, Z_train, X_train,
                epochs=epochs, lr=self.estimator_lr, M_train=self.gmmn_M_train,
                kernel_type=self.gmmn_kernel, optimizer=optimizer, 
                grad_clip=self.gmmn_grad_clip
            )
        else:
            raise ValueError(f"Unknown estimator_type: {self.estimator_type}")

    def _init_estimator(self, Z_train, mu_train, X_train):
        """
        Initialize the estimator based on mode. Called by subclass after data is prepared.
        
        Args:
        - Z_train: Conditioning variables for training (only used in pseudo_model_x mode)
        - mu_train: Ground truth mean (only used in pseudo_model_x mode, for MLP debugging)
        - X_train: Target variable for training (only used in pseudo_model_x mode)
        """
        if self.mode == MODE_MODEL_X:
            self.estimator = None
        elif self.mode == MODE_PSEUDO_MODEL_X:
            # Create the model based on estimator_type
            self.estimator = self._create_estimator_by_type(self.estimator_type)
            
            # Train on all data
            self.estimator_optimizer = self._train_estimator_by_type(
                self.estimator, Z_train, X_train, gt_mu=mu_train,
                epochs=self.estimator_epochs_pseudo
            )
            
            # For MLP with shrinkage covariance, compute global covariance from training residuals
            if self.estimator_type == ESTIMATOR_MLP and self.use_shrinkage_cov:
                self._update_global_covariance(Z_train, X_train)
        elif self.mode == MODE_ONLINE:
            self.estimator = None
            self.estimator_optimizer = None
            self.accumulated_Z = None
            self.accumulated_X = None
            self.accumulated_gt_mu = None
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'model_x', 'pseudo_model_x', 'online'.")
    
    def update_online_estimator(self, Z_new, X_new, gt_mu_new=None, epochs=None):
        """
        Update the online estimator with new training data.
        Called by the trainer after each training sequence.
        In online mode, continues training the existing model instead of training from scratch.
        
        Supports all estimator types: MLP, GMMN.
        
        Args:
        - Z_new (torch.Tensor): New Z training data (n x z_dim)
        - X_new (torch.Tensor): New X training data (n x x_dim)
        - gt_mu_new (torch.Tensor, optional): True conditional mean (n x x_dim) for debugging
        - epochs (int, optional): Number of training epochs (uses config default if not specified)
        """
        if self.mode != MODE_ONLINE:
            return
        
        if epochs is None:
            epochs = self.estimator_epochs_online
            
        # Accumulate data
        if self.accumulated_Z is None:
            self.accumulated_Z = Z_new
            self.accumulated_X = X_new
            self.accumulated_gt_mu = gt_mu_new
        else:
            self.accumulated_Z = torch.cat([self.accumulated_Z, Z_new], dim=0)
            self.accumulated_X = torch.cat([self.accumulated_X, X_new], dim=0)
            if gt_mu_new is not None and self.accumulated_gt_mu is not None:
                self.accumulated_gt_mu = torch.cat([self.accumulated_gt_mu, gt_mu_new], dim=0)
        
        # Create model if first time
        if self.estimator is None:
            self.estimator = self._create_estimator_by_type()
        
        # Train using the unified training method
        self.estimator_optimizer = self._train_estimator_by_type(
            self.estimator,
            self.accumulated_Z,
            self.accumulated_X,
            gt_mu=self.accumulated_gt_mu,
            epochs=epochs,
            optimizer=self.estimator_optimizer
        )
        
        # Update global covariance for MLP with shrinkage
        if self.estimator_type == ESTIMATOR_MLP and self.use_shrinkage_cov:
            self._update_global_covariance(self.accumulated_Z, self.accumulated_X)

    def _update_global_covariance(self, Z, X):
        """
        Update the global covariance matrix using all available data.
        
        This computes residuals from the current estimator and updates the
        global covariance estimate with shrinkage regularization.
        
        Args:
        - Z: Conditioning variables (n x z_dim)
        - X: Target variables (n x x_dim)
        """
        if self.estimator is None or self.estimator_type != ESTIMATOR_MLP:
            return
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Z_tensor = Z.to(device) if isinstance(Z, torch.Tensor) else torch.from_numpy(Z).to(torch.float32).to(device)
        X_tensor = X.to("cpu") if isinstance(X, torch.Tensor) else torch.from_numpy(X).to(torch.float32)
        
        with torch.no_grad():
            mu = self.estimator(Z_tensor).to("cpu")
            residuals = X_tensor - mu
            
            # Compute shrinkage covariance
            self._global_cov_matrix = _shrinkage_cov(residuals, alpha=self.shrinkage_alpha)
            
            # Compute Cholesky factor for efficient sampling: Sigma = L @ L.T
            # We'll sample as: X_tilde = mu + L @ eps, where eps ~ N(0, I)
            try:
                self._global_cov_cholesky = torch.linalg.cholesky(self._global_cov_matrix)
            except RuntimeError:
                # If Cholesky fails, fall back to eigendecomposition
                eigvals, eigvecs = torch.linalg.eigh(self._global_cov_matrix)
                eigvals = torch.clamp(eigvals, min=1e-8)
                self._global_cov_cholesky = eigvecs @ torch.diag(torch.sqrt(eigvals))

    def regenerate_all_tilde(self, data):
        """
        Regenerate all X_tilde values in a dataset using the updated estimator.
        
        Args:
        - data: DatasetOperator or MergedDataset to regenerate X_tilde values for
        """
        if self.estimator is None:
            return
            
        try:
            if hasattr(data, 'regenerate_tilde'):
                data.regenerate_tilde(self.estimator, self.z_dim, self.x_dim, 
                                      estimator_type=self.estimator_type,
                                      use_shrinkage_cov=self.use_shrinkage_cov,
                                      shrinkage_alpha=self.shrinkage_alpha,
                                      cov_cholesky=self._global_cov_cholesky)
        except Exception as e:
            import logging
            logging.warning(f"Failed to regenerate X_tilde: {e}")


def sample_X_tilde_given_Z_estimator(Z, X, mu_X_given_Z_estimator, use_shrinkage_cov=False, 
                                     shrinkage_alpha=0.1, cov_cholesky=None):
    """
    Sample X_tilde using an estimator for the conditional mean.
    Common function used by CIT datasets.
    
    Args:
    - Z: Conditioning variables (n x z_dim)
    - X: Original X values (n x x_dim), used to estimate residual variance
    - mu_X_given_Z_estimator: Trained estimator for E[X|Z], or None
    - use_shrinkage_cov: If True, use shrinkage covariance estimation for sampling
    - shrinkage_alpha: Shrinkage strength (only used when use_shrinkage_cov=True and cov_cholesky is None)
    - cov_cholesky: Pre-computed Cholesky factor of global covariance (d, d). If provided, uses this for sampling.
    
    Returns:
    - X_tilde: Sampled values (n x x_dim). If estimator is None, returns X directly.
    """
    X = torch.from_numpy(X).to(torch.float32) if isinstance(X, np.ndarray) else X
    
    # If no estimator yet (e.g., online mode at step=0), return X directly
    if mu_X_given_Z_estimator is None:
        return X
    
    Z = torch.from_numpy(Z).to(torch.float32) if isinstance(Z, np.ndarray) else Z
    Z = Z.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mu = mu_X_given_Z_estimator(Z).to("cpu")
        
        if cov_cholesky is not None:
            # Use pre-computed global covariance Cholesky factor
            # Sample: X_tilde = mu + L @ eps, where eps ~ N(0, I)
            n, d = mu.shape
            eps = torch.randn(n, d, device=mu.device)
            Sigma = cov_cholesky @ cov_cholesky.T
            X_tilde = mu + eps @ cov_cholesky.T
        elif use_shrinkage_cov:
            # Use batch-level shrinkage covariance estimation
            residuals = X - mu
            X_tilde = _sample_with_shrinkage_cov(mu, residuals, shrinkage_alpha)
        else:
            # Original simple diagonal variance estimation
            residuals = X - mu
            sigma_hat = torch.std(residuals, dim=0)
            epsilon = torch.randn_like(mu) * sigma_hat
            X_tilde = mu + epsilon
    return X_tilde


def _shrinkage_cov(R, alpha=0.1):
    """
    Compute shrinkage covariance estimate.
    
    Shrinks the sample covariance toward a scaled identity matrix for better
    conditioning, especially in high-dimensional settings.
    
    Args:
    - R: Residuals tensor (n, d)
    - alpha: Shrinkage strength (0 = pure sample cov, 1 = pure identity)
    
    Returns:
    - Sigma_shrunk: Shrinkage covariance matrix (d, d)
    """
    n, d = R.shape
    R_centered = R - R.mean(dim=0, keepdim=True)
    Sigma = (R_centered.T @ R_centered) / (n - 1)
    
    trace = torch.trace(Sigma)
    Sigma_shrunk = (1 - alpha) * Sigma + alpha * (trace / d) * torch.eye(d, device=R.device)
    
    return Sigma_shrunk


def _whitening_matrix(Sigma, eps=1e-8):
    """
    Compute whitening matrix W such that W @ Sigma @ W.T ≈ I.
    
    Args:
    - Sigma: Covariance matrix (d, d)
    - eps: Small value for numerical stability
    
    Returns:
    - W: Whitening matrix (d, d)
    """
    # Use eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals = torch.clamp(eigvals, min=eps)
    
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals))
    W = D_inv_sqrt @ eigvecs.T
    return W


def _sample_with_shrinkage_cov(mu, residuals, alpha=0.1):
    """
    Sample X_tilde using shrinkage covariance estimation.
    
    This method:
    1. Estimates covariance of residuals with shrinkage regularization
    2. Computes whitening matrix and its inverse
    3. Samples from N(mu, Sigma_shrunk) using the inverse whitening transform
    
    Args:
    - mu: Predicted conditional means (n, d)
    - residuals: Residuals R = X - mu (n, d)
    - alpha: Shrinkage strength
    
    Returns:
    - X_tilde: Sampled values (n, d)
    """
    n, d = mu.shape
    device = mu.device
    
    # Compute shrinkage covariance
    Sigma_hat = _shrinkage_cov(residuals, alpha=alpha)
    
    # Compute whitening matrix and its inverse
    W = _whitening_matrix(Sigma_hat)
    W_inv = torch.linalg.inv(W)
    
    # Sample from standard normal and transform
    eps = torch.randn(n, d, device=device)
    R_new = eps @ W_inv.T
    
    X_tilde = mu + R_new
    return X_tilde


def sample_X_tilde_given_Z_gmmn(Z, gmmn_estimator, x_dim=1):
    """
    Sample X_tilde using a GMMN (Generative Moment Matching Network) estimator.
    
    GMMN (Dziugaite et al., 2015; Li et al., 2015) generates samples by:
    1. Sampling noise η ~ N(0, I_m)
    2. Passing (η, z) through the trained generator G
    3. Output G(η, z) as approximately sampled from P(X|Z=z)
    
    Args:
    - Z: Conditioning variables (n x z_dim)
    - gmmn_estimator: Trained GMMN_Estimator model, or None
    
    Returns:
    - X_tilde: Sampled values (n x x_dim). If estimator is None, returns zeros.
    """
    Z = torch.from_numpy(Z).to(torch.float32) if isinstance(Z, np.ndarray) else Z
    
    # If no estimator yet (e.g., online mode at step=0), return zeros
    if gmmn_estimator is None:
        return torch.zeros(Z.shape[0], x_dim)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z = Z.to(device)
    with torch.no_grad():
        X_tilde = gmmn_estimator.sample(Z).to("cpu")
    return X_tilde


def sample_from_estimator(Z, X, estimator, estimator_type, use_shrinkage_cov=False, shrinkage_alpha=0.1, cov_cholesky=None):
    """
    Unified sampling function that dispatches to the appropriate sampling method
    based on estimator type.
    
    Args:
    - Z: Conditioning variables (n x z_dim)
    - X: Original X values (n x x_dim), used for MLP residual estimation
    - estimator: Trained estimator model (MLP or GMMN)
    - estimator_type: Type of estimator ('mlp', 'gmmn')
    - use_shrinkage_cov: If True, use shrinkage covariance for MLP sampling
    - shrinkage_alpha: Shrinkage strength (only used when use_shrinkage_cov=True)
    - cov_cholesky: Pre-computed Cholesky factor of global covariance (optional)
    
    Returns:
    - X_tilde: Sampled values (n x x_dim)
    """
    if estimator_type == ESTIMATOR_MLP:
        return sample_X_tilde_given_Z_estimator(Z, X, estimator, 
                                                 use_shrinkage_cov=use_shrinkage_cov,
                                                 shrinkage_alpha=shrinkage_alpha,
                                                 cov_cholesky=cov_cholesky)
    elif estimator_type == ESTIMATOR_GMMN:
        return sample_X_tilde_given_Z_gmmn(Z, estimator, x_dim=X.shape[1])
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}. Choose from 'mlp', 'gmmn'.")