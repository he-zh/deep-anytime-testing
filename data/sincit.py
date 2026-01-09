import numpy as np
import torch
from torch.utils.data import Dataset

from .estimate_x_given_z import train_estimator

from .datagen import DatasetOperator, DataGenerator


def get_sin_cit_data(d=20, n=100, test='type1', seed=0, beta=1.0, alpha=0.1,
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0):
    """Generate data for the SinCIT test.
    Code adapted from https://github.com/shaersh/ecrt/
    
    :param d: dimension of the data
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
    # c = np.random.RandomState(seed=seed*(n+1)).normal(0, 1, size=(n, d))
    c = np.random.normal(0, 1, size=(n, d))
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
    """Sample a new realization of 'a' given c.
    
    :param c: conditioning variables (n, d)
    :param a_m: mean of a given c (n, 1)
    :param alpha: noise scale
    :return: new sample of a
    """
    n = a_m.shape[0]
    a_r = np.random.normal(0, 1, size=(n, 1))
    a = a_m + alpha * a_r
    
    return a

def sample_a_tilde_given_c(c, a_m, a, a_given_c_estimator):
    """Sampling a_tilde ~ N(mu, sigma)
    
    :param c: conditioning variables (n, d)
    :param a_m: mean of a given c (n, 1)
    :param alpha: noise scale
    :return: new sample of a_tilde
    """
    c = c.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mu = a_given_c_estimator(c).to("cpu")
        residuals = a - mu
        sigma_hat = torch.std(residuals) # Single scalar value
        
        # 3. Generate a_tilde = mu_hat(c) + epsilon where epsilon ~ N(0, sigma_hat)
        epsilon = torch.randn_like(mu) * sigma_hat
        a_tilde = mu + epsilon
    return a_tilde

class SinCIT(DatasetOperator):
    """
    Sinusoidal Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Sinusoidal CIT dataset compatible with
    the GaussianCIT interface.
    """

    def __init__(self, type, samples, d, seed, tau1, tau2, beta=1.0, alpha=0.1, model_x=False, X_given_Z_estimator=None):
        """
        Initialize the SinCIT object.

        Args:
        - type (str): Specifies the type of dataset ('type1' or 'type2').
        - samples (int): Number of samples in the dataset.
        - d (int): Dimension of the conditioning variable c.
        - seed (int): Random seed for reproducibility.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.
        - beta (float): Parameter for type2 test correlation.
        - alpha (float): Noise scale parameter.
        """
        super().__init__(tau1, tau2)

        # Retrieve data for Sinusoidal CIT
        a, b, c, a_m = get_sin_cit_data(n=samples, d=d, test=type, seed=seed, beta=beta, alpha=alpha)
        a = torch.from_numpy(a).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        # Create a sample from a given c (similar to X_tilde in GaussianCIT)
        if model_x:
            a_tilde = sample_a_given_c(a_m, alpha)
            a_tilde = torch.from_numpy(a_tilde).to(torch.float32)
            c = torch.from_numpy(c).to(torch.float32)
        else:
            c = torch.from_numpy(c).to(torch.float32)
            a_m = torch.from_numpy(a_m).to(torch.float32)
            a_tilde = sample_a_tilde_given_c(c, a_m, a, X_given_Z_estimator).to('cpu')

        
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


class SinCITGen(DataGenerator):
    """
    Sinusoidal CIT Data Generator class that extends the DataGenerator.

    This class is responsible for generating datasets using the Sinusoidal CIT method.
    """

    def __init__(self, type, samples, d, data_seed, beta=1.0, alpha=0.1, model_x=False):
        """
        Initialize the SinCITGen object.

        Args:
        - type (str): Specifies the type of dataset ('type1' or 'type2').
        - samples (int): Number of samples to generate.
        - d (int): Dimension of conditioning variable.
        - data_seed (int): Seed for random number generation.
        - beta (float): Parameter for type2 test.
        - alpha (float): Noise scale parameter.
        - model_x (bool): Whether to model X given Z.
        """
        super().__init__(type, samples, data_seed)
        self.type = type
        self.samples = samples
        self.data_seed = data_seed
        self.d = d
        self.beta = beta
        self.alpha = alpha
        self.model_x = model_x
        if model_x:
            self.X_given_Z_estimator = None
        else:
            a_train, _, c_train, a_m_train = get_sin_cit_data(
                n=5000, d=d, test=type, seed=99999, beta=beta, alpha=alpha)
            a_train = torch.from_numpy(a_train).to(torch.float32)
            c_train = torch.from_numpy(c_train).to(torch.float32)
            a_m_train = torch.from_numpy(a_m_train).to(torch.float32)
            self.X_given_Z_estimator = train_estimator(
                Z_train=c_train, 
                gt_mu=a_m_train, 
                X_train=a_train, 
            )

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
        return SinCIT(self.type, self.samples, self.d, modified_seed, tau1, tau2, 
                      beta=self.beta, alpha=self.alpha, model_x=self.model_x, 
                      X_given_Z_estimator=self.X_given_Z_estimator)