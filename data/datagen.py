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

    def regenerate_tilde(self, mu_X_given_Z_estimator, z_dim=1):
        """
        Regenerate X_tilde using an updated estimator.
        
        This method should be called after the online estimator is updated
        to refresh all X_tilde values with the improved estimator.
        
        The data structure in self.z is:
        - z[:, :, 0] = Z = [X_target, Z_cov, Y]
        - z[:, :, 1] = Z_tilde = [X_tilde, Z_cov, Y]
        
        Where X_target is at index 0, Z_cov is at indices 1:z_dim+1, Y is at z_dim+1:
        
        Args:
        - mu_X_given_Z_estimator: Updated estimator for E[X|Z]
        - z_dim: Dimension of conditioning variable Z_cov (default 1)
        """
        if mu_X_given_Z_estimator is None:
            return
            
        # Extract from original Z (z[:, :, 0])
        Z_original = self.z[:, :, 0]
        X_target = Z_original[:, :1]              # (n, 1)
        Z_cov = Z_original[:, 1:z_dim+1]          # (n, z_dim)
        Y = Z_original[:, z_dim+1:]               # (n, 1) or more
        
        # Regenerate X_tilde with the updated estimator
        X_tilde_new = sample_X_tilde_given_Z_estimator(
            Z_cov, X_target, mu_X_given_Z_estimator
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
        
        # Will be set by subclass or during training
        self.mu_X_given_Z_estimator = None
        self.estimator_optimizer = None
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
        - model: mu_X_Given_Z_Estimator instance
        """
        from .estimate_x_given_z import mu_X_Given_Z_Estimator
        
        return mu_X_Given_Z_Estimator(
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
            self.mu_X_given_Z_estimator = None
        elif self.mode == MODE_PSEUDO_MODEL_X:
            # Create the model
            self.mu_X_given_Z_estimator = self._create_estimator(input_dim=Z_train.shape[1])
            
            # Train on all data without validation split
            self.estimator_optimizer = train_estimator(
                self.mu_X_given_Z_estimator,
                Z_train, mu_train, X_train,
                epochs=self.estimator_epochs_pseudo,
                lr=self.estimator_lr
            )
        elif self.mode == MODE_ONLINE:
            self.mu_X_given_Z_estimator = None
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
        
        Args:
        - Z_new (torch.Tensor): New Z training data (n x d-1)
        - X_new (torch.Tensor): New X training data (n x 1)
        - gt_mu_new (torch.Tensor, optional): True conditional mean (n x 1) for debugging
        - epochs (int, optional): Number of training epochs (uses config default if not specified)
        """
        if self.mode != MODE_ONLINE:
            return
        
        from .estimate_x_given_z import train_estimator
        
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
        if self.mu_X_given_Z_estimator is None:
            self.mu_X_given_Z_estimator = self._create_estimator(input_dim=self.accumulated_Z.shape[1])
        
        # Continue training existing model (gt_mu can be None for real data)
        self.estimator_optimizer = train_estimator(
            self.mu_X_given_Z_estimator,
            self.accumulated_Z, 
            self.accumulated_gt_mu,  # Can be None for real data
            self.accumulated_X,
            epochs=epochs,
            lr=self.estimator_lr,
            optimizer=self.estimator_optimizer  # Reuse optimizer state for warm start
        )

    def regenerate_all_tilde(self, data):
        """
        Regenerate all X_tilde values in a dataset using the updated estimator.
        
        Args:
        - data: DatasetOperator or MergedDataset to regenerate X_tilde values for
        """
        if self.mu_X_given_Z_estimator is None:
            return
            
        try:
            if hasattr(data, 'regenerate_tilde'):
                data.regenerate_tilde(self.mu_X_given_Z_estimator, self.z_dim)
        except Exception as e:
            import logging
            logging.warning(f"Failed to regenerate X_tilde: {e}")


def sample_X_tilde_given_Z_estimator(Z, X, mu_X_given_Z_estimator):
    """
    Sample X_tilde using an estimator for the conditional mean.
    Common function used by both GaussianCIT and SinCIT.
    
    Args:
    - Z: Conditioning variables (n x d)
    - X: Original X values (n x 1), used to estimate residual variance
    - mu_X_given_Z_estimator: Trained estimator for E[X|Z], or None
    
    Returns:
    - X_tilde: Sampled values (n x 1). If estimator is None, returns X directly.
    """
    X = torch.from_numpy(X).to(torch.float32) if isinstance(X, np.ndarray) else X
    
    # If no estimator yet (e.g., online mode at step=0), return X directly
    if mu_X_given_Z_estimator is None:
        return X
    
    Z = torch.from_numpy(Z).to(torch.float32) if isinstance(Z, np.ndarray) else Z
    Z = Z.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mu = mu_X_given_Z_estimator(Z).to("cpu")
        residuals = X - mu
        sigma_hat = torch.std(residuals)
        epsilon = torch.randn_like(mu) * sigma_hat
        X_tilde = mu + epsilon
    return X_tilde