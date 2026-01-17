"""
Car Insurance Dataset for Conditional Independence Testing.

Adapted from the kernel-ci-testing repository:
https://github.com/romanpogodin/kernel-ci-testing

The car insurance data tests whether insurance premiums (Y) are conditionally 
independent of minority status (X) given state risk (Z). This is a fairness 
testing scenario.

Data source: https://github.com/felipemaiapolo/cit/tree/main (MIT license)
"""

import numpy as np
import pandas as pd
import torch
import copy
import os
from torch.utils.data import Dataset
import scipy.stats as stats

from .datagen import (
    DatasetOperator, CITDataGeneratorBase,
    MODE_ONLINE, sample_from_estimator, ESTIMATOR_MLP
)


def data_normalize(data):
    """Normalize data using z-score normalization."""
    data = stats.zscore(data, ddof=1, axis=0)
    data[np.isnan(data)] = 0.
    return data


def find_nearest(array, value):
    """Find the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_available_states(data_path):
    """
    Get list of available states from the data directory.
    
    Args:
        data_path: Path to the directory containing CSV files
    
    Returns:
        List of state codes (e.g., ['ca', 'il', 'mo', 'tx'])
    """
    states = []
    for f in os.listdir(data_path):
        if f.endswith('-per-zip.csv'):
            state = f.replace('-per-zip.csv', '')
            states.append(state)
    return sorted(states)


def get_companies_for_state(data_path, state):
    """
    Get list of unique insurance companies for a given state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
    
    Returns:
        List of company names
    """
    csv_path = os.path.join(data_path, f'{state}-per-zip.csv')
    data = pd.read_csv(csv_path)
    data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
    companies = sorted(data['companies_name'].unique().tolist())
    return companies


def get_company_sample_size(data_path, state, company):
    """
    Get the number of samples available for a specific company in a state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code
        company: Company name
    
    Returns:
        Number of samples for that company
    """
    csv_path = os.path.join(data_path, f'{state}-per-zip.csv')
    data = pd.read_csv(csv_path)
    data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
    company_data = data.loc[data.companies_name == company]
    return len(company_data)


def get_company_by_index(data_path, state, company_idx):
    """
    Get company name by index for a given state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
        company_idx: 0-based index of the company (0, 1, 2, ...)
    
    Returns:
        Company name string
    
    Raises:
        ValueError: If company_idx is out of range
    """
    companies = get_companies_for_state(data_path, state)
    if company_idx < 0 or company_idx >= len(companies):
        raise ValueError(f"company_idx {company_idx} is out of range. "
                        f"State {state} has {len(companies)} companies (0-{len(companies)-1})")
    return companies[company_idx]


def get_num_companies(data_path, state):
    """
    Get the total number of companies for a state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
    
    Returns:
        Number of companies in that state
    """
    return len(get_companies_for_state(data_path, state))


def load_car_insurance_full(data_path, state='ca', n_vals=20, test='type1', 
                            data_seed=0, company=None, verbose=False):
    """
    Load and process the full car insurance data for a given data_seed.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
        n_vals: Number of bins for discretizing state_risk
        test: 'type1' for H0 (simulated independence), 'type2' for real data (H1)
        data_seed: Seed for shuffling Y within Z bins (defines the dataset)
        company: Optional company name filter
        verbose: If True, print available companies
    
    Returns:
        a: Combined premium (Y) - what we're testing
        b: Minority status (X) - binary
        c: State risk (Z) - conditioning variable
    """
    # Load data
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    csv_path = os.path.join(data_path, f'{state}-per-zip.csv')
    data = pd.read_csv(csv_path)
    data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
    
    if verbose:
        print(f"Available companies in {state}: {data['companies_name'].unique().tolist()}")
    
    if company is not None:
        data = data.loc[data.companies_name == company]
    
    Z = np.array(data.state_risk).reshape((-1, 1))
    Y = np.array(data.combined_premium).reshape((-1, 1))
    X = (1 * np.array(data.minority)).reshape((-1, 1))  # Binary minority status
    
    if test == 'type1':
        # Simulated H0: shuffle Y within Z bins to break X-Y dependence given Z
        bins = np.linspace(np.min(Z), np.max(Z), n_vals + 2)
        bins = bins[1:-1]
        Y_ci = copy.deepcopy(Y)
        Z_bin = np.array([find_nearest(bins, z) for z in Z.squeeze()]).reshape(Z.shape)
        
        # Use data_seed to define the shuffling (this defines the full dataset)
        for val in np.unique(Z_bin):
            ind = Z_bin == val
            ind2 = np.random.choice(np.sum(ind), np.sum(ind), replace=False)
            Y_ci[ind] = Y_ci[ind][ind2]
        
        # Use binned Z for conditioning
        c = Z_bin
        a = Y_ci
        b = X
    else:
        # Real data (type2) - test actual conditional independence
        c = Z
        a = Y
        b = X
    
    return a, b, c


class CarInsuranceCIT(DatasetOperator):
    """
    Car Insurance Conditional Independence Test dataset.
    
    Tests: Y (premium) ⊥ X (minority) | Z (state risk)
    
    For the car insurance data, we use online estimation of E[a|c] to generate
    a_tilde, similar to the GaussianCIT and SinCIT datasets.
    
    Variables:
    - a: Premium (Y) - continuous
    - b: Minority status (X) - binary 
    - c: State risk (Z) - conditioning variable
    
    The test structure is: [a, c, b] where we estimate a|c and sample a_tilde.
    """

    def __init__(self, a, b, c, tau1, tau2, z_dim=1, x_dim=1, estimator=None, estimator_type=ESTIMATOR_MLP):
        """
        Initialize the CarInsuranceCIT object from pre-loaded data arrays.

        Args:
            a: Premium tensor (Y) - shape (n, 1)
            b: Minority status tensor (X) - binary - shape (n, 1)
            c: State risk tensor (Z) - shape (n, 1)
            tau1: First transformation operator
            tau2: Second transformation operator
            estimator: Trained estimator for E[a|c], or None for online mode
            estimator_type: Type of estimator ('mlp', 'gmmn')
        """
        super().__init__(tau1, tau2, z_dim=z_dim, x_dim=x_dim)
        
        # No ground truth conditional mean for real data
        self.ground_truth_mu = None

        # Generate a_tilde using the estimator (online mode)
        # a_tilde is sampled from estimated distribution of a|c
        a_tilde = sample_from_estimator(c, a, estimator, estimator_type).to('cpu')
        
        # Construct X and Y for the test
        # X = [a, c], Y = b
        X = torch.cat((a, c), dim=1)
        Y = b
        X_tilde = torch.cat((a_tilde, c), dim=1)
        
        # Z = [X, Y] = [a, c, b]
        Z = torch.cat((X, Y), dim=1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1)
        
        # Stack the two tensors along a new dimension
        self.z = torch.stack([Z, Z_tilde], dim=2)

    @classmethod
    def from_datasets(cls, datasets, tau1=None, tau2=None):
        """Combine multiple CarInsuranceCIT datasets."""
        combined = cls.__new__(cls)

        combined.tau1 = tau1 if tau1 is not None else datasets[0].tau1
        combined.tau2 = tau2 if tau2 is not None else datasets[0].tau2
        
        # Reconstruct z by concatenating from individual datasets
        combined.z = torch.cat([d.z for d in datasets], dim=0)
        
        return combined


class CarInsuranceCITGen(CITDataGeneratorBase):
    """
    Car Insurance CIT Data Generator.
    
    Generates datasets for testing conditional independence in car insurance data.
    Loads the full dataset for the given data_seed and samples without
    replacement across sequences.
    
    This generator only supports online mode since we don't have access to
    the true conditional distribution E[a|c]. The estimator is trained
    incrementally as data accumulates across sequences.
    """

    def __init__(self, type, samples, data_seed, data_path, mode=MODE_ONLINE, state='ca', n_vals=20, 
                 company=None, company_idx=None, normalize=True, verbose=False,
                 estimator_cfg=None):
        """
        Initialize the CarInsuranceCITGen object.

        Args:
            type: 'type1' for simulated H0, 'type2' for real data
            samples: Number of samples per batch
            data_seed: Seed for defining the shuffled dataset (Y permuted within Z bins)
            data_path: Path to data directory containing state CSV files
            state: State code ('ca', 'il', 'mo', 'tx')
            n_vals: Number of bins for discretizing state_risk
            company: Optional company name filter (use this OR company_idx)
            company_idx: Optional 0-based company index (0, 1, 2, ...). 
                        Use get_num_companies(data_path, state) to get total count.
            normalize: If True, normalize the data using z-score
            verbose: If True, print debug information
            estimator_cfg: Config dict for the online estimator model
        """
        # Initialize base class with online mode (the only supported mode)
        super().__init__(type, samples, data_seed, mode=MODE_ONLINE, 
                         pretrain_samples=0, estimator_cfg=estimator_cfg)
        
        self.data_path = data_path
        self.state = state
        self.n_vals = n_vals
        self.normalize = normalize
        self.verbose = verbose
        
        # Resolve company from company_idx if provided
        if company_idx is not None:
            company = get_company_by_index(data_path, state, company_idx)
        self.company = company
        
        # Load the full dataset for this data_seed
        a_np, b_np, c_np = load_car_insurance_full(
            data_path=data_path,
            state=state,
            n_vals=n_vals,
            test=type,
            data_seed=data_seed,
            company=company,
            verbose=verbose
        )
        
        # Normalize continuous variables if requested
        if normalize:
            a_np = data_normalize(a_np)
            c_np = data_normalize(c_np)
        
        # Store full data as tensors
        self.full_a = torch.tensor(a_np, dtype=torch.float32)
        self.full_b = torch.tensor(b_np, dtype=torch.float32)
        self.full_c = torch.tensor(c_np, dtype=torch.float32)
        self.max_n_points = len(self.full_a)
        
        # Data dimension: a(1) + c(1) + b(1) = 3
        self.d = 3
        
        # Z dimension for the estimator (c dimension = 1)
        self._z_dim = 1
        # X dimension for the estimator (a dimension = 1)
        self._x_dim = 1

        # Initialize available indices for non-replacement sampling
        self.available_indices = list(range(self.max_n_points))
        
        # Set seeds
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        
        # Shuffle the available indices once based on data_seed
        self.rng = np.random.default_rng(seed=data_seed)
        self.rng.shuffle(self.available_indices)
        self.current_idx = 0  # Pointer to track where we are in the shuffled indices
        
        # Initialize online estimator (starts as None, created on first update)
        self._init_estimator(None, None, None)
        
        if verbose:
            print(f"CarInsuranceCITGen initialized: {self.max_n_points} total samples, "
                  f"state={state}, company={company}, mode=online")


    def get_remaining_samples(self):
        """Get the number of remaining samples available for sampling."""
        return self.max_n_points - self.current_idx

    def generate(self, seed, tau1, tau2, samples=None) -> CarInsuranceCIT:
        """
        Generate data by sampling without replacement from the loaded dataset.

        Args:
            seed: Not used for sampling (kept for API compatibility)
            tau1: First transformation operator
            tau2: Second transformation operator
            samples: Optional override for number of samples

        Returns:
            Dataset: A CarInsuranceCIT dataset
        """
        samples = self.samples if samples is None else samples
        modified_seed = (self.data_seed + 1) * 1000 + seed
        torch.manual_seed(modified_seed)
        np.random.seed(modified_seed)
        
        # Check if we have enough samples left
        if self.current_idx + samples > self.max_n_points:
            raise ValueError(
                f"Not enough samples left for non-replacement sampling. "
                f"Requested {samples}, but only {self.max_n_points - self.current_idx} remaining. "
                f"Consider using fewer sequences or smaller batch sizes."
            )
        
        # Get the next batch of indices (without replacement)
        batch_indices = self.available_indices[self.current_idx:self.current_idx + samples]
        self.current_idx += samples
        
        # Extract data for these indices
        a = self.full_a[batch_indices]
        b = self.full_b[batch_indices]
        c = self.full_c[batch_indices]
        
        return CarInsuranceCIT(a, b, c, tau1, tau2, z_dim=self._z_dim, x_dim=self._x_dim,
                               estimator=self.estimator, estimator_type=self.estimator_type)
