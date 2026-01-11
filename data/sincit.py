import numpy as np
import torch
from torch.utils.data import Dataset

from .datagen import (
    DatasetOperator, CITDataGeneratorBase,
    MODE_MODEL_X, MODE_PSEUDO_MODEL_X, MODE_ONLINE,
    sample_X_tilde_given_Z_estimator
)


def get_sin_cit_data(z_dim=20, n=100, test='type1', seed=0, beta=1.0, alpha=0.1,
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0):
    """Generate data for the SinCIT test.
    Code adapted from https://github.com/shaersh/ecrt/
    
    :param z_dim: dimension of conditioning variable Z (c)
    :param n: number of samples
    :param test: type of the test ('type1' or 'type2')
    :param seed: random seed
    :param beta: parameter for the type 2 test
    :param alpha: noise scale parameter
    :param ca_dim_idx: dimension index for variable a
    :param cb_dim_idx: dimension index for variable b
    :param cr_dim_idx: dimension index for correlation in type2
    :return: a, b, c data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    c = np.random.normal(0, 1, size=(n, z_dim))
    f = np.cos
    g = np.exp
    a_m = f(c[:, ca_dim_idx:ca_dim_idx+1])
    b_m = g(c[:, cb_dim_idx:cb_dim_idx+1])

    if test == 'type2':
        r = np.sin(beta * c[:, cr_dim_idx])
        a_r = np.zeros((n, 1))
        b_r = np.zeros((n, 1))
        for i in range(n):
            cov_matrix = [[1, r[i]], [r[i], 1]]
            # a_r[i, 0], b_r[i, 0] = np.random.RandomState(seed=seed*(n+1)+1+i).multivariate_normal([0, 0], cov_matrix)
            a_r[i, 0], b_r[i, 0] = np.random.multivariate_normal([0, 0], cov_matrix)
    elif test == 'type1':
        # a_r = np.random.RandomState(seed=seed*(n+1)+1).normal(0, 1, size=(n, 1))
        # b_r = np.random.RandomState(seed=seed*(n+1)+2).normal(0, 1, size=(n, 1))
        a_r = np.random.normal(0, 1, size=(n, 1))
        b_r = np.random.normal(0, 1, size=(n, 1))
    else:
        raise NotImplementedError(f'{test} has to be type1 or type2')
    a = a_m + alpha * a_r
    b = b_m + alpha * b_r

    
    return a, b, c, a_m


def sample_a_given_c(a_m, alpha):
    """Sample a new realization of 'a' given c using the true conditional mean (ModelX setting).
    
    :param a_m: mean of a given c (n, 1)
    :param alpha: noise scale
    :return: new sample of a
    """
    n = a_m.shape[0]
    a_r = np.random.normal(0, 1, size=(n, 1))
    a = a_m + alpha * a_r
    return a


class SinCIT(DatasetOperator):
    """
    Sinusoidal Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Sinusoidal CIT dataset compatible with
    the GaussianCIT interface.
    """

    def __init__(self, type, samples, z_dim, seed, tau1, tau2, beta=1.0, alpha=0.1, 
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0,
                 mode=MODE_MODEL_X, X_given_Z_estimator=None):
        """
        Initialize the SinCIT object.

        Args:
        - type (str): Specifies the type of dataset ('type1' or 'type2').
        - samples (int): Number of samples in the dataset.
        - z_dim (int): Dimension of conditioning variable Z (c).
        - seed (int): Random seed for reproducibility.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.
        - beta (float): Parameter for type2 test correlation.
        - alpha (float): Noise scale parameter.
        - mode (str): Mode for X|Z estimation. One of MODE_MODEL_X, MODE_PSEUDO_MODEL_X, MODE_ONLINE.
        - X_given_Z_estimator: Pretrained estimator for a given c.
        """
        super().__init__(tau1, tau2)

        # Retrieve data for Sinusoidal CIT
        a, b, c, a_m = get_sin_cit_data(n=samples, z_dim=z_dim, test=type, seed=seed, beta=beta, alpha=alpha,
                                         ca_dim_idx=ca_dim_idx, cb_dim_idx=cb_dim_idx, cr_dim_idx=cr_dim_idx)

        # Create a sample from a given c based on mode
        if mode == MODE_MODEL_X:
            # Use true μ to sample ã
            a_tilde = sample_a_given_c(a_m, alpha)
            a_tilde = torch.from_numpy(a_tilde).to(torch.float32)
        else:
            # Use estimator (pseudo_model_x or online mode)
            a_tilde = sample_X_tilde_given_Z_estimator(c, a, X_given_Z_estimator).to('cpu')

        # Convert numpy arrays to torch tensors
        a = torch.from_numpy(a).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        c = torch.from_numpy(c).to(torch.float32)
        
        # Construct X and Y
        # X = [a, c], Y = b
        X = torch.cat((a, c), dim=1)
        Y = b
        X_tilde = torch.cat((a_tilde, c), dim=1)

        # Concatenate tensors: Z = [X, Y]
        Z = torch.cat((X, Y), dim=1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1)

        # Stack the two tensors along a new dimension
        self.z = torch.stack([Z, Z_tilde], dim=2)


class SinCITGen(CITDataGeneratorBase):
    """
    Sinusoidal CIT Data Generator class that extends CITDataGeneratorBase.

    This class is responsible for generating datasets using the Sinusoidal CIT method.
    Supports three modes:
    - model_x: Use true μ to sample ã (oracle setting)
    - pseudo_model_x: Pre-train estimator with extra data at initialization
    - online: Estimate μ using accumulated training data, updated each sequence
    """

    def __init__(self, type, samples, z_dim, data_seed, beta=1.0, alpha=0.1,
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0,
                 mode=MODE_MODEL_X, pretrain_samples=5000, estimator_cfg=None):
        """
        Initialize the SinCITGen object.

        Args:
        - type (str): Specifies the type of dataset ('type1' or 'type2').
        - samples (int): Number of samples to generate.
        - z_dim (int): Dimension of conditioning variable Z (c).
        - data_seed (int): Seed for random number generation.
        - beta (float): Parameter for type2 test.
        - alpha (float): Noise scale parameter.
        - ca_dim_idx (int): Dimension index for variable a.
        - cb_dim_idx (int): Dimension index for variable b.
        - cr_dim_idx (int): Dimension index for correlation in type2.
        - mode (str): Mode for X|Z estimation. One of 'model_x', 'pseudo_model_x', 'online'.
        - pretrain_samples (int): Number of samples for pre-training estimator (only used in pseudo_model_x mode).
        - estimator_cfg (dict, optional): Config for estimator model.
        """
        super().__init__(type, samples, data_seed, mode, pretrain_samples, estimator_cfg)
        self._z_dim = z_dim
        self.d = z_dim + 1 + 1  # Total dimension: a(1) + c(z_dim) + b(1)
        self.beta = beta
        self.alpha = alpha
        self.ca_dim_idx = ca_dim_idx
        self.cb_dim_idx = cb_dim_idx
        self.cr_dim_idx = cr_dim_idx
        
        # Initialize estimator based on mode
        self._initialize_mode(mode, pretrain_samples, type)
    
    @property
    def z_dim(self):
        """Dimension of conditioning variable Z (c)."""
        return self._z_dim
    
    def _initialize_mode(self, mode, pretrain_samples, type):
        """Initialize estimator based on mode."""
        if mode == MODE_PSEUDO_MODEL_X:
            a_train, _, c_train, a_m_train = get_sin_cit_data(
                n=pretrain_samples, z_dim=self._z_dim, test=type, seed=99999+self.data_seed, 
                beta=self.beta, alpha=self.alpha,
                ca_dim_idx=self.ca_dim_idx, cb_dim_idx=self.cb_dim_idx, cr_dim_idx=self.cr_dim_idx)
            a_train = torch.from_numpy(a_train).to(torch.float32)
            c_train = torch.from_numpy(c_train).to(torch.float32)
            a_m_train = torch.from_numpy(a_m_train).to(torch.float32)
            self._init_estimator(c_train, a_m_train, a_train)
        else:
            self._init_estimator(None, None, None)

    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate data using the SinCIT method.

        Args:
        - seed (int): Seed for random number generation.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.

        Returns:
        - Dataset: A dataset generated using SinCIT.
        """
        # Use a modified seed value based on the provided seed and class's data_seed
        modified_seed = (self.data_seed + 1) * 100 + seed
        return SinCIT(self.type, self.samples, self._z_dim, modified_seed, tau1, tau2, 
                      beta=self.beta, alpha=self.alpha,
                      ca_dim_idx=self.ca_dim_idx, cb_dim_idx=self.cb_dim_idx, cr_dim_idx=self.cr_dim_idx,
                      mode=self.mode, X_given_Z_estimator=self.X_given_Z_estimator)