from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import numpy as np

# Mode constants for X|Z estimation
MODE_MODEL_X = "model_x"           # Use true μ to sample X̃
MODE_PSEUDO_MODEL_X = "pseudo_model_x"  # Pre-train estimator with extra data
MODE_ONLINE = "online"             # Estimate μ online using training data


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


class CITDataGeneratorBase(DataGenerator):
    """
    Base class for Conditional Independence Test data generators.
    Handles the three modes: model_x, pseudo_model_x, and online.
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
        self.estimator_patience = self.estimator_cfg.get('patience', 10)
        self.estimator_epochs = self.estimator_cfg.get('epochs', 500)
        self.estimator_val_ratio = self.estimator_cfg.get('val_ratio', 0.2)  # For pseudo_model_x train/val split
        
        # Will be set by subclass or during training
        self.X_given_Z_estimator = None
        self.estimator_optimizer = None
        self.estimator_early_stopper = None
        self.accumulated_Z = None
        self.accumulated_X = None
        self.accumulated_gt_mu = None
    
    @property
    def z_dim(self):
        """
        Return the dimension of the conditioning variable Z.
        Must be implemented by subclasses.
        
        Returns:
        - int: Dimension of Z (the conditioning variable for X|Z estimation)
        """
        raise NotImplementedError("Subclasses must implement z_dim property")
        
    def _create_estimator(self, input_dim):
        """
        Create a new estimator model with the configured architecture.
        
        Args:
        - input_dim: Input dimension (Z dimension)
        
        Returns:
        - model: PX_Given_Z_Estimator instance
        """
        from .estimate_x_given_z import PX_Given_Z_Estimator
        
        return PX_Given_Z_Estimator(
            input_dim=input_dim,
            hidden_size=self.estimator_hidden_size,
            layer_norm=self.estimator_layer_norm,
            drop_out=self.estimator_drop_out,
            drop_out_p=self.estimator_drop_out_p
        )
    
    def _init_estimator(self, Z_train, mu_train, X_train):
        """
        Initialize the estimator based on mode. Called by subclass after data is prepared.
        
        Args:
        - Z_train: Conditioning variables for training (only used in pseudo_model_x mode)
        - mu_train: Ground truth mean (only used in pseudo_model_x mode)
        - X_train: Target variable for training (only used in pseudo_model_x mode)
        """
        from .estimate_x_given_z import train_estimator
        
        if self.mode == MODE_MODEL_X:
            self.X_given_Z_estimator = None
        elif self.mode == MODE_PSEUDO_MODEL_X:
            # Create the model
            self.X_given_Z_estimator = self._create_estimator(input_dim=Z_train.shape[1])
            
            # Split data into train/val for early stopping
            n = len(Z_train)
            n_val = int(n * self.estimator_val_ratio)
            indices = torch.randperm(n)
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]
            
            Z_tr, Z_val = Z_train[train_indices], Z_train[val_indices]
            mu_tr, mu_val = mu_train[train_indices], mu_train[val_indices]
            X_tr, X_val = X_train[train_indices], X_train[val_indices]
            
            self.estimator_optimizer, self.estimator_early_stopper = train_estimator(
                self.X_given_Z_estimator,
                Z_tr, mu_tr, X_tr,
                Z_val=Z_val, X_val=X_val,
                epochs=self.estimator_epochs,
                patience=500,
                lr=self.estimator_lr
            )
        elif self.mode == MODE_ONLINE:
            self.X_given_Z_estimator = None
            self.estimator_optimizer = None
            self.estimator_early_stopper = None
            self.accumulated_Z = None
            self.accumulated_X = None
            self.accumulated_gt_mu = None
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'model_x', 'pseudo_model_x', 'online'.")
    
    def update_online_estimator(self, Z_new, X_new, gt_mu_new=None, Z_val=None, X_val=None, epochs=None):
        """
        Update the online estimator with new training data.
        Called by the trainer after each training sequence.
        In online mode, continues training the existing model instead of training from scratch.
        
        Args:
        - Z_new (torch.Tensor): New Z training data (n x d-1)
        - X_new (torch.Tensor): New X training data (n x 1)
        - gt_mu_new (torch.Tensor, optional): True conditional mean (n x 1) for debugging
        - Z_val (torch.Tensor, optional): Validation Z data for early stopping
        - X_val (torch.Tensor, optional): Validation X data for early stopping
        - epochs (int, optional): Number of training epochs (uses config default if not specified)
        """
        if self.mode != MODE_ONLINE:
            return
        
        from .estimate_x_given_z import train_estimator
        
        if epochs is None:
            epochs = self.estimator_epochs
            
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
        
        # Use true gt_mu if available (for debugging), otherwise use zeros
        if self.accumulated_gt_mu is not None:
            gt_mu = self.accumulated_gt_mu
        else:
            gt_mu = torch.zeros_like(self.accumulated_X)
        
        # Create model if first time
        if self.X_given_Z_estimator is None:
            self.X_given_Z_estimator = self._create_estimator(input_dim=self.accumulated_Z.shape[1])
        
        # Continue training existing model
        self.estimator_optimizer, self.estimator_early_stopper = train_estimator(
            self.X_given_Z_estimator,
            self.accumulated_Z, 
            gt_mu,
            self.accumulated_X,
            Z_val=Z_val,
            X_val=X_val,
            epochs=epochs,
            patience=self.estimator_patience,
            lr=self.estimator_lr,
            optimizer=self.estimator_optimizer,  # Reuse optimizer state
            early_stopper=self.estimator_early_stopper
        )


def sample_X_tilde_given_Z_estimator(Z, X, X_given_Z_estimator):
    """
    Sample X_tilde using an estimator for the conditional mean.
    Common function used by both GaussianCIT and SinCIT.
    
    Args:
    - Z: Conditioning variables (n x d)
    - X: Original X values (n x 1), used to estimate residual variance
    - X_given_Z_estimator: Trained estimator for E[X|Z], or None
    
    Returns:
    - X_tilde: Sampled values (n x 1). If estimator is None, returns X directly.
    """
    X = torch.from_numpy(X).to(torch.float32) if isinstance(X, np.ndarray) else X
    
    # If no estimator yet (e.g., online mode at step=0), return X directly
    if X_given_Z_estimator is None:
        return X
    
    Z = torch.from_numpy(Z).to(torch.float32) if isinstance(Z, np.ndarray) else Z
    Z = Z.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mu = X_given_Z_estimator(Z).to("cpu")
        residuals = X - mu
        sigma_hat = torch.std(residuals)
        epsilon = torch.randn_like(mu) * sigma_hat
        X_tilde = mu + epsilon
    return X_tilde