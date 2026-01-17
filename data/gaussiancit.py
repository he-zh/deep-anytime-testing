import numpy as np
import torch
from torch.utils.data import Dataset
from .datagen import (
    DatasetOperator, CITDataGeneratorBase, 
    MODE_MODEL_X, MODE_PSEUDO_MODEL_X, MODE_ONLINE,
    sample_from_estimator, ESTIMATOR_MLP
)


def get_cit_data(z_dim=19, a=3, n=5000, test='type1', seed=0, u=None, v=None):
    """Generate data for the PCR test.
     Code from https://github.com/shaersh/ecrt/
    :param z_dim: dimension of conditioning variable Z
    :param a: parameter for the type 2 test
    :param n: number of samples
    :param test: type of the test
    :return: X, Y data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z_mu = np.zeros((z_dim, 1)).ravel()
    Z_Sigma = np.eye(z_dim)
    Z = np.random.multivariate_normal(Z_mu, Z_Sigma, n) # Z is n x z_dim
    if v is None: v = np.random.normal(0, 1, (z_dim, 1))
    X_mu = Z @ v
    X = np.random.normal(X_mu, 1, (n, 1)) # X is n x 1
    if u is None: u = np.random.normal(0, 1, (z_dim, 1))
    if test == 'type2':
        Y_mu = (Z @ u) ** 2 + a * X
    elif test == 'type1':
        Y_mu = (Z @ u) ** 2
    Y = np.random.normal(Y_mu, 1, (n, 1)) # Y is n x 1
    X = np.column_stack((X, Z)) # X is n x (z_dim+1)
    return X, Y, X_mu

def sample_X_given_Z(X_mu):
    """Sample X given Z using the true conditional mean (ModelX setting)."""
    n, d = X_mu.shape
    X = np.random.normal(X_mu, 1, (n, 1))
    # X = np.column_stack((X, Z))
    return X


class GaussianCIT(DatasetOperator):
    """
    Gaussian Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Gaussian CIT dataset.
    """

    def __init__(self, type, samples, seed, tau1, tau2, u, v, z_dim, x_dim=1, mode=MODE_MODEL_X, 
                 estimator=None, estimator_type=ESTIMATOR_MLP):
        """
        Initialize the GaussianCIT object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples in the dataset.
        - seed (int): Random seed for reproducibility.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.
        - u (numpy.ndarray): A parameter for generating CIT data.
        - v (numpy.ndarray): Another parameter for generating CIT data.
        - z_dim (int): Dimension of conditioning variable Z.
        - x_dim (int): Dimension of target variable X.
        - mode (str): Mode for X|Z estimation. One of MODE_MODEL_X, MODE_PSEUDO_MODEL_X, MODE_ONLINE.
        - estimator: Pretrained estimator for X given Z.
        - estimator_type: Type of estimator ('mlp', 'gmmn').
        """
        super().__init__(tau1, tau2, z_dim=z_dim, x_dim=x_dim)
        # Retrieve data for Gaussian CIT
        X, Y, mu = get_cit_data(z_dim=z_dim, u=u, v=v, n=samples, test=type, seed=seed)

        # Convert numpy arrays to PyTorch tensors
        X = torch.from_numpy(X).to(torch.float32)
        Y = torch.from_numpy(Y).to(torch.float32)
        
        # Store ground truth conditional mean for monitoring estimation error
        self.ground_truth_mu = torch.from_numpy(mu).to(torch.float32)

        # Create a sample from X given Z based on mode
        if mode == MODE_MODEL_X:
            # Use true μ to sample X̃
            X_tilde = sample_X_given_Z(mu) # X_tilde is n x d, (X_tilde[:,0] is sampled from X|Z)
            X_tilde = torch.from_numpy(X_tilde).to(torch.float32)
        else:
            # Use estimator (pseudo_model_x or online mode)
            Z_cov = X[:, 1:]  # Conditioning variables
            X_target = X[:, :1]  # Target variable
            X_tilde = sample_from_estimator(Z_cov, X_target, estimator, estimator_type).to('cpu')
        
        
        X_tilde = torch.cat((X_tilde, X[:, 1:]), dim=1)

        # Concatenate tensors
        Z = torch.cat((X, Y), dim=1) # Z is n x (d+1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1) # Z_tilde is n x (d+1)

        # Stack the two tensors along a new dimension
        self.z = torch.stack([Z, Z_tilde], dim=2) # z is n x (d+1) x 2


class GaussianCITGen(CITDataGeneratorBase):
    """
    Gaussian CIT Data Generator class that extends CITDataGeneratorBase.

    This class is responsible for generating datasets using the GaussianCIT method.
    Supports three modes:
    - model_x: Use true μ to sample X̃ (oracle setting)
    - pseudo_model_x: Pre-train estimator with extra data at initialization
    - online: Estimate μ using accumulated training data, updated each sequence
    """

    def __init__(self, type, samples, data_seed, z_dim=19, mode=MODE_MODEL_X, pretrain_samples=5000,
                 estimator_cfg=None):
        """
        Initialize the GaussianCITGen object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate per batch.
        - data_seed (int): Seed for random number generation.
        - z_dim (int): Dimension of conditioning variable Z.
        - mode (str): Mode for X|Z estimation. One of 'model_x', 'pseudo_model_x', 'online'.
        - pretrain_samples (int): Number of samples for pre-training estimator (only used in pseudo_model_x mode).
        - estimator_cfg (dict, optional): Config for estimator model.
        """
        # Initialize base class first (don't call _init_estimator yet)
        super().__init__(type, samples, data_seed, mode, pretrain_samples, estimator_cfg)
        self._z_dim = z_dim
        self._x_dim = 1
        self.d = z_dim + 1 + 1  # Total dimension: X(1) + Z(z_dim) + Y(1)
        
        # Generate random vectors for u and v
        self.v = np.random.normal(0, 1, (self._z_dim, 1))
        self.u = np.random.normal(0, 1, (self._z_dim, 1))
        
        # Initialize estimator based on mode
        self._initialize_mode(mode, pretrain_samples, type)
        
    def _initialize_mode(self, mode, pretrain_samples, type):
        """Initialize estimator based on mode."""
        if mode == MODE_PSEUDO_MODEL_X:
            X_train, _, mu_train = get_cit_data(z_dim=self._z_dim, u=self.u, v=self.v, 
                                                 n=pretrain_samples, test=type, seed=999999+self.data_seed)
            X_train = torch.from_numpy(X_train).to(torch.float32)
            mu_train = torch.from_numpy(mu_train).to(torch.float32)
            X_val, Z_train = X_train[:, :1], X_train[:, 1:]
            self._init_estimator(Z_train, mu_train, X_val)
        else:
            self._init_estimator(None, None, None)

    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate data using the GaussianCIT method.

        Args:
        - seed (int): Seed for random number generation.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.

        Returns:
        - Dataset: A dataset generated using GaussianCIT.
        """
        # Use a modified seed value based on the provided seed and class's data_seed
        modified_seed = (self.data_seed + 1) * 1000 + seed
        return GaussianCIT(self.type, self.samples, modified_seed, tau1, tau2, self.u, self.v,
                          z_dim=self._z_dim, x_dim=self._x_dim, mode=self.mode, estimator=self.estimator)
