from trainer import Trainer
import numpy as np
import torch


class TrainerECRT(Trainer):
    """Trainer class for the ECRT method."""

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """
        Initializes the TrainerECRT object by extending the Trainer class.

        Args:
        (same as Trainer class)
        """
        super().__init__(cfg, net, tau1, tau2, datagen, device, data_seed)
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.integral_vector = torch.tensor(np.linspace(0, 1, 1001, endpoint=False)[1:]).to(device)
        self.batches = [2, 5, 10]
        self.stb = [torch.ones(1000, device=device) for _ in range(len(self.batches))]


    def ecrt(self, y, x, x_tilde, mode):
        """
        Implement the ECRT method.
        Code adapted from https://github.com/shaersh/ecrt/

        Args:
        - y (torch.Tensor): Ground truth labels.
        - x (torch.Tensor): Input samples.
        - x_tilde (torch.Tensor): Perturbed samples.
        - mode (str): Either "train" or "test".

        Returns:
        - torch.Tensor: Test statistic.
        """
        self.net.eval()
        total_samples = y.shape[0]
        test_stat = torch.nn.MSELoss(reduction='mean')
        st = []
        stb = [torch.ones(1000, device=self.device) for _ in range(len(self.batches))]
        i = 0

        for b in self.batches:
            # split data into batches of size b
            num_chunks = int(total_samples / b)
            index_sets_seq = np.array_split(range(total_samples), num_chunks)
            stb.append(torch.ones(1000, device=self.device))
            for ind in index_sets_seq:
                y_tb, x_tb, x_tilde_tb = y[ind], x[ind], x_tilde[ind]
                pred_tilde_tb = self.net(x_tilde_tb).detach()
                pred_tb = self.net(x_tb).detach()

                q = test_stat(pred_tb, y_tb)
                q_tilde = test_stat(pred_tilde_tb, y_tb)
                wealth = torch.nn.Tanh()(q_tilde - q)

                if mode == "test":
                    self.stb[i] = self.stb[i] * (1 + self.integral_vector * wealth)
                    stb[i] = self.stb[i].clone()
                else:
                    stb[i] = stb[i] * (1 + self.integral_vector * wealth)

            st.append(stb[i].mean())
            i += 1
        st = torch.stack(st).mean()
        return st

    def train_evaluate_epoch(self, loader, mode="train"):
        """
        Train/Evaluate the model for one epoch using the ECRT approach.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train" or "test".

        Returns:
        - tuple: Aggregated loss and E-value for the current epoch.
        """
        aggregated_loss, loss_tilde = 0, 0
        num_samples = len(loader.dataset)

        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            target_size = loader.dataset.x_dim
            features = z[:, :-target_size]
            target = z[:, -target_size:]
            tau_z = tau_z.to(self.device)

            self.net.train() if mode == "train" else self.net.eval()

            out = self.net(features)
            loss = self.loss(out, target)
            aggregated_loss += loss

            y_tilde = self.net(tau_z[:, :-target_size]).detach()
            loss_tilde += self.loss(y_tilde, target).item()

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            e_val = self.ecrt(target, features, tau_z[:, :-target_size], mode)

        self.log({
            f"{mode}_loss": aggregated_loss.item() / (i + 1),
            f"{mode}_e-val": e_val.item(),
            f"{mode}_loss_tilde": loss_tilde / (i + 1)
        })
        return aggregated_loss / num_samples, e_val