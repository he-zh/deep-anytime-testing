import numpy as np
import torch
from torch.utils.data import Dataset
from .estimate_x_given_z import train_estimator
from .datagen import DatasetOperator, DataGenerator


def get_cit_data(d=20, a=3, n=5000, test='type1', seed=0, u=None, v=None):
    """Generate data for the PCR test.
     Code from https://github.com/shaersh/ecrt/
    :param d: dimension of the data
    :param a: parameter for the type 2 test
    :param n: number of samples
    :param test: type of the test
    :return: X, Y data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z_mu = np.zeros((d-1, 1)).ravel()
    Z_Sigma = np.eye(d-1)
    Z = np.random.multivariate_normal(Z_mu, Z_Sigma, n) # Z is n x (d-1)
    if v is None: v = np.random.normal(0, 1, (d-1, 1))
    X_mu = Z @ v
    X = np.random.normal(X_mu, 1, (n, 1)) # X is n x 1
    if u is None: u = np.random.normal(0, 1, (d-1, 1))
    beta = np.ones((d, 1))
    if test == 'type2':
        Y_mu = (Z @ u) ** 2 + a * X
    elif test == 'type1':
        Y_mu = (Z @ u) ** 2
        beta[0] = 0
    Y = np.random.normal(Y_mu, 1, (n, 1)) # Y is n x 1
    X = np.column_stack((X, Z)) # X is n x d
    return X, Y, X_mu

def sample_X_given_Z(Z, X_mu):
    n, d = Z.shape
    X = np.random.normal(X_mu, 1, (n, 1))
    X = np.column_stack((X, Z))
    return X


def sample_X_tilde_given_Z(Z, gt_mu, X, X_given_Z_estimator):
    # Sampling X_tilde ~ N(mu, sigma)
    Z = Z.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mu = X_given_Z_estimator(Z).to("cpu")
        residuals = X - mu
        sigma_hat = torch.std(residuals) # Single scalar value
        
        # 3. Generate X_tilde = mu_hat(Z) + epsilon where epsilon ~ N(0, sigma_hat)
        epsilon = torch.randn_like(mu) * sigma_hat
        X_tilde = mu + epsilon

    X_tilde = torch.cat((X_tilde, Z.to("cpu")), dim=1)
    return X_tilde


class GaussianCIT(DatasetOperator):
    """
    Gaussian Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Gaussian CIT dataset.
    """

    def __init__(self, type, samples, seed, tau1, tau2, u, v, model_x=False, X_given_Z_estimator=None):
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
        - X_given_Z_estimator (PX_Given_Z_Estimator, optional): Pretrained estimator for X given Z.
        """
        super().__init__(tau1, tau2)

        # Retrieve data for Gaussian CIT
        X, Y, mu = get_cit_data(u=u, v=v, n=samples, test=type, seed=seed)


        # Create a sample from X given Z
        if model_x:
            X_tilde = sample_X_given_Z(X[:, 1:], mu) # X_tilde is n x d, (X_tilde[:,0] is sampled from X|Z)
            X_tilde = torch.from_numpy(X_tilde).to(torch.float32)
            X = torch.from_numpy(X).to(torch.float32)
        else:
            X = torch.from_numpy(X).to(torch.float32)
            mu = torch.from_numpy(mu).to(torch.float32)
            X_tilde = sample_X_tilde_given_Z(X[:, 1:], mu, X[:, :1], X_given_Z_estimator).to('cpu')

        # Convert numpy arrays to PyTorch tensors
        Y = torch.from_numpy(Y).to(torch.float32)

        # Concatenate tensors
        Z = torch.cat((X, Y), dim=1) # Z is n x (d+1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1) # Z_tilde is n x (d+1)

        # Stack the two tensors along a new dimension
        self.z = torch.stack([Z, Z_tilde], dim=2) # z is n x (d+1) x 2


class GaussianCITGen(DataGenerator):
    """
    Gaussian CIT Data Generator class that extends the DataGenerator.

    This class is responsible for generating datasets using the GaussianCIT method.
    """

    def __init__(self, type, samples, data_seed, model_x=False):
        """
        Initialize the GaussianCITGen object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate.
        - data_seed (int): Seed for random number generation.
        """
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed = type, samples, data_seed
        self.d = 20 # for now is hardcoded
        # Generate random vectors for u and v
        self.v = np.random.normal(0, 1, (self.d - 1, 1))
        self.u = np.random.normal(0, 1, (self.d - 1, 1))
        self.model_x = model_x
        # Train the estimator for X given Z
        if model_x:
            self.X_given_Z_estimator = None
        else:
            X_train, _, mu_train = get_cit_data(u=self.u, v=self.v, n=5000, test=type, seed=999999)
            X_train = torch.from_numpy(X_train).to(torch.float32)
            mu_train = torch.from_numpy(mu_train).to(torch.float32)
            X_train, Z_train = X_train[:, :1], X_train[:, 1:]
            self.X_given_Z_estimator = train_estimator(Z_train, mu_train, X_train)


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
        modified_seed = (self.data_seed + 1) * 100 + seed
        return GaussianCIT(self.type, self.samples, modified_seed, tau1, tau2, self.u, self.v, model_x=self.model_x, X_given_Z_estimator=self.X_given_Z_estimator)
