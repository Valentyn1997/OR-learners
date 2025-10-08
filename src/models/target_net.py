from pyro.nn import DenseNN
import logging
import torch
from omegaconf import DictConfig
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import MLFlowLogger
from src.models.base_net import BaseNet
from torch_ema import ExponentialMovingAverage
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TargetNet(BaseNet):

    val_metric = None
    name = 'target_net'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, target=None, **kwargs):

        super(TargetNet, self).__init__(args, mlflow_logger)

        self.target = target
        assert target in ['cate', 'mu0', 'mu1', 'y0', 'y1', 'rcate', 'ivw_pi_cate', 'ivw_a_cate']

        self.out_scaler = StandardScaler()

        self.target_inp = args.exp.target_inp
        if self.target_inp == 'repr':
            self.dim_input = args.repr_net.dim_repr
        elif self.target_inp in ['cov', 'cov_repr']:
            self.dim_input = self.dim_cov
        elif self.target_inp == 'out':
            self.dim_input = 2
        else:
            raise NotImplementedError()
        self.gamma = args.target_net.gamma
        self.ema_target = None
        self.hid_layers = args.target_net.hid_layers
        self.q_trunc = args.repr_net.q_trunc

        if self.target_inp != 'cov_repr':
            self.dim_hid1 = int(args.target_net.dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_input)
            self.nn = DenseNN(self.dim_input, self.hid_layers * [self.dim_hid1], param_dims=[1], nonlinearity=torch.nn.ELU()).float()
        else:
            self.dim_hid1 = int(args.repr_net.dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_input)
            self.dim_repr = args.repr_net.dim_repr
            self.dim_hid2 = int(args.target_net.dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_repr)
            self.nn = DenseNN(self.dim_input,
                              self.hid_layers * [self.dim_hid1] + [self.dim_repr] + self.hid_layers * [self.dim_hid2],
                              param_dims=[1], nonlinearity=torch.nn.ELU()).float()

        self.to(self.device)

    def prepare_train_data(self, data_dict: dict):
        self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'])

        # Torch tensors
        if self.target_inp == 'repr':
            inp_f = torch.tensor(data_dict['repr_f']).float()
        elif self.target_inp in ['cov', 'cov_repr']:
            inp_f = torch.tensor(cov_f).float()
        elif self.target_inp == 'out':
            mu0_pred_cov = torch.tensor(data_dict['mu_pred0_cov'].reshape(-1, 1)).float()
            mu1_pred_cov = torch.tensor(data_dict['mu_pred1_cov'].reshape(-1, 1)).float()
            inp_f = torch.cat([mu0_pred_cov, mu1_pred_cov], dim=1)
        else:
            raise NotImplementedError()

        self.hparams.dataset.n_samples_train = inp_f.shape[0]

        self.num_train_iter = int(self.hparams.dataset.n_samples_train / self.batch_size * self.num_epochs)
        logger.info(f'Effective number of training iterations: {self.num_train_iter}.')

        if 'prop_pred_cov' in data_dict:
            prop_pred = torch.tensor(data_dict['prop_pred_cov'].reshape(-1, 1)).float()
        else:
            prop_pred = torch.tensor(data_dict['prop_pred_repr'].reshape(-1, 1)).float()

        prop_pred = 1 - prop_pred if self.target == 'y0' else prop_pred

        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))
        _, treat_f, out_f = self.prepare_tensors(None, data_dict['treat_f'], out_f)

        if 'mu_pred0_cov' in data_dict:
            mu0_pred = torch.tensor(data_dict['mu_pred0_cov'].reshape(-1, 1)).float()
            mu1_pred = torch.tensor(data_dict['mu_pred1_cov'].reshape(-1, 1)).float()
        else:
            mu0_pred = torch.tensor(data_dict['mu_pred0_repr'].reshape(-1, 1)).float()
            mu1_pred = torch.tensor(data_dict['mu_pred1_repr'].reshape(-1, 1)).float()

        mu_pred = mu0_pred if self.target == 'y0' else mu1_pred

        if self.target in ['cate', 'mu0', 'mu1', 'ivw_pi_cate', 'ivw_a_cate']:
            pseudo_out = torch.tensor(data_dict[f'pseudo_{self.target}']).float()
            return inp_f, pseudo_out, treat_f, out_f, prop_pred, mu0_pred, mu1_pred
        elif self.target in ['y0', 'y1', 'rcate']:
            if self.target in ['y0', 'y1']:
                return inp_f, treat_f, out_f, prop_pred, mu_pred
            else:
                return inp_f, treat_f, out_f, prop_pred, mu0_pred, mu1_pred
        else:
            raise NotImplementedError()

    def prepare_eval_data(self, data_dict: dict):
        self.cov_scaler.fit_transform(data_dict['cov_f'])
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'])

        # Torch tensors
        if self.target_inp == 'repr':
            inp_f = torch.tensor(data_dict['repr_f']).float()
        elif self.target_inp in ['cov', 'cov_repr']:
            inp_f = torch.tensor(cov_f).float()
        elif self.target_inp == 'out':
            mu0_pred_cov = torch.tensor(data_dict['mu_pred0_cov'].reshape(-1, 1)).float()
            mu1_pred_cov = torch.tensor(data_dict['mu_pred1_cov'].reshape(-1, 1)).float()
            inp_f = torch.cat([mu0_pred_cov, mu1_pred_cov], dim=1)
        else:
            raise NotImplementedError()

        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        out_pot0 = torch.tensor(out_pot0).reshape(-1, 1).float()
        out_pot1 = torch.tensor(out_pot1).reshape(-1, 1).float()
        mu0 = torch.tensor(mu0).reshape(-1, 1).float() if self.oracle_available else None
        mu1 = torch.tensor(mu1).reshape(-1, 1).float() if self.oracle_available else None

        return inp_f, out_pot0, out_pot1, mu0, mu1

    def get_train_dataloader(self, *tensors):
        training_data = TensorDataset(*tensors)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        ema_target = ExponentialMovingAverage(self.parameters(), decay=self.gamma)
        return optimizer, ema_target

    def fit(self, train_data_dict: dict, log: bool, kind='pseudo'):
        if self.target in ['cate', 'mu0', 'mu1', 'ivw_pi_cate', 'ivw_a_cate']:
            inp_f, pseudo_out, treat_f, out_f, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = self.prepare_train_data(train_data_dict)
            train_dataloader = self.get_train_dataloader(inp_f, pseudo_out, treat_f, out_f, prop_pred_cov, mu0_pred_cov, mu1_pred_cov)
        elif self.target in ['y0', 'y1']:
            inp_f, treat_f, out_f, prop_pred_cov, mu_pred_cov = self.prepare_train_data(train_data_dict)
            train_dataloader = self.get_train_dataloader(inp_f, treat_f, out_f, prop_pred_cov, mu_pred_cov)
        elif self.target in ['rcate']:
            inp_f, treat_f, out_f, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = self.prepare_train_data(train_data_dict)
            train_dataloader = self.get_train_dataloader(inp_f, treat_f, out_f, prop_pred_cov, mu0_pred_cov, mu1_pred_cov)
        else:
            raise NotImplementedError()

        optimizer, self.ema_target = self.get_optimizer()

        for step in tqdm(range(self.num_train_iter)) if log else range(self.num_train_iter):
            if self.target in ['cate', 'mu0', 'mu1', 'ivw_pi_cate', 'ivw_a_cate']:
                inp_f, pseudo_out, treat_f, out_f, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = next(iter(train_dataloader))
            elif self.target in ['y0', 'y1']:
                inp_f, treat_f, out_f, prop_pred_cov, mu_pred_cov = next(iter(train_dataloader))
            elif self.target in ['rcate']:
                inp_f, treat_f, out_f, prop_pred_cov, mu0_pred_cov, mu1_pred_cov = next(iter(train_dataloader))
            else:
                raise NotImplementedError()
            optimizer.zero_grad()

            pred = self.nn(inp_f)

            if self.target in ['cate', 'mu0', 'mu1']:

                loss = ((pred - pseudo_out) ** 2).mean()
                log_dict = {f'train_mse_target': loss.item()}
            elif self.target in ['y0', 'y1']:
                inter = 0.0 if self.target == 'y0' else 1.0
                ipwt = ((treat_f == inter) & (prop_pred_cov >= self.q_trunc)).float() / (prop_pred_cov + 1e-9)
                loss = (ipwt * (pred - out_f) ** 2 + (1.0 - ipwt) * (mu_pred_cov - out_f) ** 2).mean()
                log_dict = {f'train_mse_cf_target': loss.item()}

            elif self.target in ['rcate']:
                mu_pred_cov = prop_pred_cov * mu1_pred_cov + (1 - prop_pred_cov) * mu0_pred_cov
                loss = (((out_f - mu_pred_cov) - (treat_f - prop_pred_cov) * pred) ** 2).mean()
                log_dict = {f'train_rloss_target': loss.item()}
            elif self.target in ['ivw_a_cate']:
                overlap_eff = (treat_f - prop_pred_cov) ** 2
                loss = (overlap_eff * (pred - pseudo_out) ** 2).mean()
                log_dict = {f'train_wmse_target': loss.item()}
            elif self.target in ['ivw_pi_cate']:  # Not Neyman-orthogonal!!
                overlap = prop_pred_cov * (1 - prop_pred_cov)
                loss = (overlap * (pred - pseudo_out) ** 2).mean()
                log_dict = {f'train_wmse_target': loss.item()}
            else:
                raise NotImplementedError()
            loss.backward()

            optimizer.step()
            self.ema_target.update()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

    def evaluate_pot_mse(self, data_dict: dict, log: bool, prefix: str, target: str):
        assert self.target == target

        inp_f, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)
        mu = mu0 if target in ['mu0', 'y0'] else mu1
        out_pot = out_pot0 if target in ['mu0', 'y0'] else out_pot1

        self.eval()
        with torch.no_grad():
            with self.ema_target.average_parameters():
                pseudo_out_pred = self.nn(inp_f)

            mse = (((pseudo_out_pred - out_pot)) ** 2).mean().sqrt()
            mse_oracle = (((mu - out_pot)) ** 2).mean().sqrt() if self.oracle_available else None

        results = {f'{prefix}_mse_{target}': mse.item(), f'{prefix}_mse_{target}_oracle': mse_oracle.item()}
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str, target: str):
        assert self.target == target

        inp_f, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            with self.ema_target.average_parameters():
                pseudo_out_pred = self.nn(inp_f)

            rpehe = ((pseudo_out_pred - (out_pot1 - out_pot0)) ** 2).mean().sqrt()
            rpehe_oracle = (((mu1 - mu0) - (out_pot1 - out_pot0)) ** 2).mean().sqrt() if self.oracle_available else None

        results = {f'{prefix}_{target}_rpehe': rpehe.item(), f'{prefix}_{target}_rpehe_oracle': rpehe_oracle.item()}
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
        return results
