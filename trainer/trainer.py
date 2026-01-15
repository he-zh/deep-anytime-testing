import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
import wandb
from models import EarlyStopper

# Import mode constant and MergedDataset for checking online mode
try:
    from data.datagen import MODE_ONLINE, MergedDataset
except ImportError:
    MODE_ONLINE = "online"
    MergedDataset = None

class Trainer:

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """
        Initializes the Trainer object with the provided configurations and parameters.

        Args:
        - cfg (Config): Configuration object containing trainer settings.
        - net (nn.Module): The neural network model to train.
        - tau1 (float): Operator 1.
        - tau2 (float): Operator 2.
        - datagen (DataGenerator): Object to generate data.
        - device (torch.device): The device (CPU/GPU) where training should take place.
        - data_seed (int): Seed for generating data.
        """
        self.data_seed = data_seed
        # Extract configurations from the cfg object
        self.seed = cfg.seed
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.seqs = cfg.seqs
        self.patience = cfg.earlystopping.patience
        self.delta = cfg.earlystopping.delta
        self.alpha = cfg.alpha
        self.T = cfg.T
        self.bs = cfg.batch_size
        self.save = cfg.save
        self.save_dir = cfg.save_dir

        # Operators
        self.tau1 = tau1
        self.tau2 = tau2

        # Model, data generator, and device assignment
        self.net = net
        self.datagen = datagen
        self.device = device

        # L1 and L2 regularization parameters
        self.weight_decay = cfg.l2_lambda
        self.l1_lambda = cfg.l1_lambda

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)

        # Variables to keep track of the current sequence and epoch
        self.current_seq = 0
        self.current_epoch = 0

    def log(self, logs):
        """
        Log metrics for visualization and monitoring.

        Args:
        - logs (dict): Dictionary containing metrics to be logged.
        """
        for key, value in logs.items():
            wandb.log({key: value}, step=self.current_seq * self.epochs + self.current_epoch)
            logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, {key}: {value}")

    def l1_regularization(self):
        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.net.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
        return l1_regularization

    def train_evaluate_epoch(self, loader, mode="train"):
        """
        Train or evaluate the model for one epoch and log the results.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train", "val", or "test". Determines how to run the model.

        Returns:
        - tuple: Aggregated loss and davt for the current epoch.
        """
        aggregated_loss = 0
        davt = 1
        num_samples = len(loader.dataset)

        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            tau_z = tau_z.to(self.device)
            if mode == "train":
                self.net.train()
                out = self.net(z, tau_z)
            else:
                self.net.eval()
                out = self.net(z, tau_z).detach()
            loss = -out.mean() + self.l1_lambda * self.l1_regularization()
            aggregated_loss += -out.sum()
            davt *= torch.exp(out.sum())
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.log({f"{mode}_e-value": davt.item(), f"{mode}_loss": aggregated_loss.item() / num_samples})
        return aggregated_loss / num_samples, davt

    def load_data(self, seed, mode="train"):
        """
        Load data using the datagen object and return a DataLoader object.

        Args:
        - seed (int): Seed for generating data.
        - mode (str): Determines how data should be loaded. Either "train", "val", or "test".

        Returns:
        - tuple: Generated data and corresponding DataLoader object.
        """
        data = self.datagen.generate(seed, self.tau1, self.tau2)
        if mode in ["train", "val"]:
            data_loader = DataLoader(data, batch_size=self.bs, shuffle=True)
        else:
            data_loader = DataLoader(data, batch_size=len(data), shuffle=True)
        return data, data_loader

    def train(self):
        """
        Train the model for a specified number of sequences and epochs, and apply early stopping if required.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_data, train_loader = self.load_data(self.seed, mode="train")
        val_data, val_loader = self.load_data(self.seed + 1, mode="val")
        davts = []
        reject_null = 0.0
        
        # For online mode: update estimator with initial training data, use val for early stopping
        self._update_online_estimator_if_needed(train_data)
        # Regenerate X_tilde for all data with updated estimator (only in online mode)
        self._regenerate_tilde_if_online(train_data)
        self._regenerate_tilde_if_online(val_data)
        train_loader = DataLoader(train_data, batch_size=self.bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.bs, shuffle=True)
        
        for k in range(self.seqs):
            self.current_seq = k
            for t in range(self.epochs):
                self.current_epoch = t
                self.train_evaluate_epoch(train_loader)
                loss_val, _ = self.train_evaluate_epoch(val_loader, mode='val')

                # Check for early stopping or end of epochs
                if self.early_stopper.early_stop(loss_val.detach()) or (t + 1) == self.epochs:

                    test_data, test_loader = self.load_data(self.seed + k + 2, mode="test")
                    _, conditional_davt = self.train_evaluate_epoch(test_loader, mode='test')
                    davts.append(conditional_davt.item())
                    davt = np.prod(np.array(davts[self.T:])) if k >= self.T else 1
                    self.log({"aggregated_test_e-value": davt})
                    train_data = MergedDataset([train_data, val_data])
                    self.log({"historical_sample_nums": len(train_data)})
                    self.log({"all_sample_nums": len(train_data) + len(test_data)})

                    # Update online estimator with new training data for the next sequence
                    self._update_online_estimator_if_needed(val_data)
                    val_data = test_data
                    
                    # Regenerate X_tilde for ALL accumulated data with updated estimator (only in online mode)
                    self._regenerate_tilde_if_online(train_data)
                    self._regenerate_tilde_if_online(val_data)

                    
                    train_loader = DataLoader(train_data, batch_size=self.bs, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=self.bs, shuffle=True)
                    break

            # Reset the early stopper for the next sequence
            self.early_stopper.reset()

            # Log information if davt exceeds the threshold
            if davt > (1. / self.alpha):
                logging.info("Reject null at %f", davt)
                self.log({"steps": k})
                reject_null = 1.0
                self.log({"reject_null": reject_null})
            else:
                self.log({"reject_null": reject_null})

    def _regenerate_tilde_if_online(self, data):
        """
        Regenerate X_tilde values only if in online mode.
        
        In pseudo_model_x mode, the estimator is fixed after pre-training,
        so there's no need to regenerate X_tilde.
        
        Args:
        - data: Dataset to regenerate X_tilde for
        """
        if not hasattr(self.datagen, 'mode') or self.datagen.mode != MODE_ONLINE:
            return
        self.datagen.regenerate_all_tilde(data)

    def _update_online_estimator_if_needed(self, train_data):
        """
        Update the online estimator if the datagen supports online mode.
        After updating, regenerates all X_tilde values in accumulated datasets.
        
        Args:
        - train_data: Dataset containing training data to accumulate for estimator
        """
        # Check if datagen has online mode and update method
        if not hasattr(self.datagen, 'mode') or self.datagen.mode != MODE_ONLINE:
            return
        if not hasattr(self.datagen, 'update_online_estimator'):
            return
            
        # Extract X and Z from the dataset
        # The data structure is: z = (X, Z_cov, Y) concatenated, shape (n, total_dim, 2)
        # where X is at indices 0:x_dim, Z_cov is at indices x_dim:x_dim+z_dim
        try:
            # Helper to extract tensors and ground_truth_mu from dataset
            def extract_from_data(data):
                if hasattr(data, 'z'):
                    z_tensor = data.z[:, :, 0]
                    mu_tensor = data.ground_truth_mu if hasattr(data, 'ground_truth_mu') else None
                    return z_tensor, mu_tensor
                return None, None
            
            z_tensor, gt_mu = extract_from_data(train_data)
            if z_tensor is None:
                return
            
            # Get dimensions from datagen
            z_dim = self.datagen.z_dim
            x_dim = self.datagen.x_dim if hasattr(self.datagen, 'x_dim') else 1
                
            # z_tensor shape: (n, total_dim) where total_dim = X_target(x_dim) + Z_cov(z_dim) + Y(y_dim)
            # X_target is at indices 0:x_dim, Z_cov is at indices x_dim:x_dim+z_dim
            X_train = z_tensor[:, :x_dim]  # (n, x_dim) - target variable
            Z_train = z_tensor[:, x_dim:x_dim+z_dim]  # (n, z_dim) - conditioning variables
            
            self.datagen.update_online_estimator(Z_train, X_train, gt_mu_new=gt_mu)
            logging.info(f"Updated online estimator with {len(z_tensor)} train samples")

        except Exception as e:
            logging.warning(f"Failed to update online estimator: {e}")
