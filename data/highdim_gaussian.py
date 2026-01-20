import numpy as np
import torch
from torch.utils.data import Dataset
from .datagen import (
    DatasetOperator, CITDataGeneratorBase, 
    MODE_MODEL_X, MODE_PSEUDO_MODEL_X, MODE_ONLINE,
    sample_from_estimator, ESTIMATOR_MLP
)

# ... (get_highdim_cit_data and sample_highdim_X_given_Z remain unchanged) ...
# I will strictly repost the get_highdim_cit_data for context if you need it, 
# but here is the focus on the Class updates.

def get_highdim_cit_data(z_dim=19, x_dim=100, a=3, n=5000, test='type1', seed=0, u=None, V=None):
    """
    Generate High-Dimensional Data.
    X|Z is Gaussian with tridiagonal covariance.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate Z
    Z = np.random.normal(0, 1, (n, z_dim))
    
    # 2. Define Conditional Mean of X (Linear: X_mu = Z @ V)
    if V is None: 
        V = np.random.normal(0, 1/np.sqrt(z_dim), (z_dim, x_dim))
    X_mu = Z @ V
    
    # 3. Construct the Residual Covariance (Tridiagonal)
    Sigma = 0.5*np.eye(x_dim) + 0.5*np.ones((x_dim, x_dim))
    
    # Compute Cholesky L such that LL^T = Sigma for fast sampling
    L = np.linalg.cholesky(Sigma)
    
    # 4. Sample X = X_mu + Noise @ L.T
    noise = np.random.normal(0, 1, (n, x_dim))
    X = X_mu + (noise @ L.T)

    # 5. Generate Y
    if u is None: 
        u = np.random.normal(0, 1, (z_dim, 1))
    
    # Base signal from Z
    y_base = (Z @ u) ** 2
    
    if test == 'type2':
        # Dependence on X: scaled by 'a'
        Y_mu = y_base + a * X
    elif test == 'type1':
        Y_mu = y_base
        Y_mu = np.tile(y_base, (1, x_dim))
        
    Y = np.random.normal(Y_mu, 1, (n, Y_mu.shape[1]))

    # X_combined is n x (x_dim + z_dim)
    X_combined = np.column_stack((X, Z)) 
    
    return X_combined, Y, X_mu, L


def sample_highdim_X_given_Z(X_mu, L):
    """Sample X given Z using the true conditional mean and precomputed Cholesky factor."""
    n, x_dim = X_mu.shape
    noise = np.random.normal(0, 1, (n, x_dim))
    X_sample = X_mu + (noise @ L.T)
    return X_sample


class HighDimGaussianCIT(DatasetOperator):
    """
    High-Dimensional Gaussian CIT dataset.
    Handles x_dim > 1 dynamically and supports covariance sampling.
    """

    def __init__(self, type, samples, seed, tau1, tau2, u, V, z_dim, x_dim=100, 
                 mode=MODE_MODEL_X, estimator=None, estimator_type=ESTIMATOR_MLP,
                 use_shrinkage_cov=False, shrinkage_alpha=0.0, cov_cholesky=None):
        """
        Args:
            ...
            use_shrinkage_cov: Whether to use shrinkage covariance for MLP sampling.
            shrinkage_alpha: Shrinkage strength for covariance.
            cov_cholesky: Pre-computed Cholesky factor of global covariance.
        """
        super().__init__(tau1, tau2, z_dim=z_dim, x_dim=x_dim)
        self.x_dim = x_dim

        # Retrieve data
        # Note: X_combined here contains [Target_X, Covariates_Z]
        X_combined, Y, mu, L = get_highdim_cit_data(
            z_dim=z_dim, x_dim=x_dim, u=u, V=V, n=samples, test=type, seed=seed
        )

        X_combined = torch.from_numpy(X_combined).to(torch.float32)
        Y = torch.from_numpy(Y).to(torch.float32)
        
        # Split Target and Covariates
        # X_target: [:, :x_dim], Z_cov: [:, x_dim:]
        X_target_true = X_combined[:, :self.x_dim]
        Z_cov = X_combined[:, self.x_dim:]

        if mode == MODE_MODEL_X:
            # Oracle: Sample from true conditional distribution
            X_tilde = sample_highdim_X_given_Z(mu, L)
            X_tilde = torch.from_numpy(X_tilde).to(torch.float32)
        else:
            # Estimated: Sample from learned model using covariance settings
            # This allows the MLP estimator to use the global covariance structure of the residuals
            X_tilde = sample_from_estimator(
                Z_cov, X_target_true, estimator, estimator_type,
                use_shrinkage_cov=use_shrinkage_cov,
                shrinkage_alpha=shrinkage_alpha,
                cov_cholesky=cov_cholesky
            ).to('cpu')
            loss = torch.sum((torch.from_numpy(L)-cov_cholesky)**2) if cov_cholesky is not None else torch.tensor(0.0)
            print(f"Covariance Cholesky Estimation Loss: {loss.item()}")
        
        # Reconstruct the "X" part of the Z vector (Target + Covariates)
        # X_combined_tilde = [X_tilde, Z]
        X_combined_tilde = torch.cat((X_tilde, Z_cov), dim=1)

        # Construct Z vectors: [X_combined, Y]
        Z_vec = torch.cat((X_combined, Y), dim=1)             # Original
        Z_tilde_vec = torch.cat((X_combined_tilde, Y), dim=1) # Resampled X

        # Stack: n x (x_dim + z_dim + 1) x 2
        self.z = torch.stack([Z_vec, Z_tilde_vec], dim=2)


class HighDimGaussianCITGen(CITDataGeneratorBase):
    """
    Generator for High Dimensional Gaussian Data with Covariance Support.
    """

    def __init__(self, type, samples, data_seed, z_dim=19, x_dim=100, mode=MODE_MODEL_X, 
                 pretrain_samples=5000, estimator_cfg=None):
        
        super().__init__(type, samples, data_seed, mode, pretrain_samples, estimator_cfg)
        self._z_dim = z_dim
        self._x_dim = x_dim
        self.d = z_dim + x_dim + 1
        
        # Generate random weights for U (impact on Y) and V (impact on X)
        np.random.seed(data_seed)
        self.V = np.random.normal(0, 1/np.sqrt(self._z_dim), (self._z_dim, self._x_dim))
        self.u = np.random.normal(0, 1, (self._z_dim, 1))
        
        # Initialize estimator
        self._initialize_mode(mode, pretrain_samples, type)
        
    def _initialize_mode(self, mode, pretrain_samples, type):
        if mode == MODE_PSEUDO_MODEL_X:
            X_train, _, mu_train, _ = get_highdim_cit_data(
                z_dim=self._z_dim, x_dim=self._x_dim, u=self.u, V=self.V, 
                n=pretrain_samples, test=type, seed=999999 + self.data_seed
            )
            X_train = torch.from_numpy(X_train).to(torch.float32)
            mu_train = torch.from_numpy(mu_train).to(torch.float32)
            
            # Slice correctly using _x_dim
            X_val_target = X_train[:, :self._x_dim]
            Z_train_cov = X_train[:, self._x_dim:]
            
            self._init_estimator(Z_train_cov, mu_train, X_val_target)
        else:
            self._init_estimator(None, None, None)

    def generate(self, seed, tau1, tau2) -> Dataset:
        modified_seed = (self.data_seed + 1) * 1000 + seed
        
        return HighDimGaussianCIT(
            self.type, self.samples, modified_seed, tau1, tau2, 
            self.u, self.V, 
            z_dim=self._z_dim, 
            x_dim=self._x_dim,
            mode=self.mode, 
            estimator=self.estimator,
            # Pass covariance arguments inherited from CITDataGeneratorBase
            estimator_type=self.estimator_type,
            use_shrinkage_cov=self.use_shrinkage_cov,
            shrinkage_alpha=self.shrinkage_alpha,
            cov_cholesky=self._global_cov_cholesky
        )