import xgboost as xgb
import logging
from sklearn.metrics import log_loss
import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.loggers import MLFlowLogger
from src.classic_models.base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostProp(BaseModel):

    val_metric = 'val_bce'
    name = 'prop_model_cov'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super().__init__(args, mlflow_logger)
        self.prop_model = xgb.XGBClassifier(n_estimators=args[self.name]['n_estimators'],
                                            max_depth=args[self.name]['max_depth'],
                                            gamma=args[self.name]['gamma'],
                                            reg_alpha=args[self.name]['reg_alpha'],
                                            min_child_weight=args[self.name]['min_child_weight'])

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))

        self.hparams.dataset.n_samples_train = cov_f.shape[0]

        return cov_f, data_dict['treat_f']

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        return cov_f, data_dict['treat_f']

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f = self.prepare_train_data(train_data_dict)
        self.prop_model.fit(cov_f, treat_f.astype(int))

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        cov_f, treat_f = self.prepare_eval_data(data_dict)

        log_dict = {f'{prefix}_bce': log_loss(treat_f.astype(int), self.prop_model.predict_proba(cov_f))}

        if log:
            self.mlflow_logger.log_metrics(log_dict, step=0)

        return log_dict

    def get_prop_predictions(self, data_dict):
        inp_f, _ = self.prepare_eval_data(data_dict)
        return self.prop_model.predict_proba(inp_f)[:, 1:]


class XGBoostMu(BaseModel):

    val_metric = 'val_rmse'
    name = 'mu_model_cov'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super().__init__(args, mlflow_logger)

        self.out_scaler = StandardScaler()

        self.mu_model_type = args.mu_model_cov.mu_model_type
        self.q_trunc = args.target_model.q_trunc

        # Mu model init
        if self.mu_model_type == 't':
            self.mu_model = [xgb.XGBRegressor(n_estimators=args[self.name]['n_estimators'],
                                              max_depth=args[self.name]['max_depth'],
                                              gamma=args[self.name]['gamma'],
                                              reg_alpha=args[self.name]['reg_alpha'],
                                              min_child_weight=args[self.name]['min_child_weight']) for _ in self.treat_options]
        elif self.mu_model_type == 's':
            self.mu_model = xgb.XGBRegressor(n_estimators=args[self.name]['n_estimators'],
                                             max_depth=args[self.name]['max_depth'],
                                             gamma=args[self.name]['gamma'],
                                             reg_alpha=args[self.name]['reg_alpha'],
                                             min_child_weight=args[self.name]['min_child_weight'])
        else:
            raise NotImplementedError()

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        self.hparams.dataset.n_samples_train = cov_f.shape[0]

        return cov_f, data_dict['treat_f'].reshape(-1, 1), out_f

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.transform(data_dict['out_f'].reshape(-1, 1))
        return cov_f, data_dict['treat_f'].reshape(-1, 1), out_f

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        if self.mu_model_type == 't':
            for i, treat in enumerate(self.treat_options):
                self.mu_model[i].fit(cov_f[treat_f.flatten() == treat], out_f[treat_f.flatten() == treat])
        elif self.mu_model_type == 's':
            inp_f = np.concatenate([cov_f, treat_f], axis=1)
            self.mu_model.fit(inp_f, out_f)
        else:
            raise NotImplementedError()


    def predict(self, cov_f, treat_f):
        pred = np.empty_like(treat_f)
        if self.mu_model_type == 't':
            for i, treat in enumerate(self.treat_options):
                pred[treat_f.flatten() == treat] = self.mu_model[i].predict(cov_f[treat_f.flatten() == treat]).reshape(-1, 1)
        elif self.mu_model_type == 's':
            inp_f = np.concatenate([cov_f, treat_f], axis=1)
            pred = self.mu_model.predict(inp_f).reshape(-1, 1)
        else:
            raise NotImplementedError()
        return pred

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        cov_f, treat_f, out_f = self.prepare_eval_data(data_dict)

        log_dict = {
            f'{prefix}_rmse': np.sqrt(((self.predict(cov_f, treat_f) - out_f) ** 2).mean())
        }

        if log:
            self.mlflow_logger.log_metrics(log_dict, step=0)

        return log_dict

    def get_outcomes(self, data_dict):
        cov_f, _, _ = self.prepare_eval_data(data_dict)

        if self.mu_model_type == 't':
            mu_pred_0, mu_pred_1 = self.mu_model[0].predict(cov_f), self.mu_model[1].predict(cov_f)
        elif self.mu_model_type == 's':
            context_0 = np.concatenate([cov_f, np.zeros((cov_f.shape[0], 1))], axis=1)
            context_1 = np.concatenate([cov_f, np.ones((cov_f.shape[0], 1))], axis=1)
            mu_pred_0, mu_pred_1 = self.mu_model.predict(context_0), self.mu_model.predict(context_1)
        else:
            raise NotImplementedError()
        return mu_pred_0.reshape(-1, 1), mu_pred_1.reshape(-1, 1)

    def get_pseudo_out(self, data_dict, target):
        cov_f, treat_f, out_f = self.prepare_eval_data(data_dict)

        mu_pred0, mu_pred1 = data_dict['mu_pred0_cov'], data_dict['mu_pred1_cov']
        prop_pred = data_dict['prop_pred_cov']

        ipwt0 = ((treat_f == 0.0) & ((1 - prop_pred) >= self.q_trunc)) / (1 - prop_pred + 1e-9)
        ipwt1 = ((treat_f == 1.0) & (prop_pred >= self.q_trunc)) / (prop_pred + 1e-9)

        if target == 'mu0':
            pseudo_out = ipwt0 * (out_f - mu_pred0) + mu_pred0
            weights = np.ones_like(treat_f)
        elif target == 'mu1':
            pseudo_out = ipwt1 * (out_f - mu_pred1) + mu_pred1
            weights = np.ones_like(treat_f)
        elif target in ['cate', 'ivw_pi_cate', 'ivw_a_cate']:
            pseudo_out = ipwt1 * (out_f - mu_pred1) + mu_pred1 - (ipwt0 * (out_f - mu_pred0) + mu_pred0)
            if target in ['cate']:
                weights = np.ones_like(treat_f)
            elif target in ['ivw_pi_cate']:
                weights = prop_pred * (1 - prop_pred)
            else:
                weights = (treat_f - prop_pred) ** 2
        elif target in ['rcate']:
            mu_pred = prop_pred * mu_pred1 + (1 - prop_pred) * mu_pred0
            pseudo_out = (out_f - mu_pred) / (treat_f - prop_pred + 1e-10)
            weights = (treat_f - prop_pred) ** 2
        else:
            raise NotImplementedError()

        return pseudo_out, weights




class XGBoostTarget(BaseModel):

    val_metric = None
    name = 'target_model'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, target=None, **kwargs):
        super().__init__(args, mlflow_logger)

        self.q_trunc = args.target_model.q_trunc
        self.out_scaler = StandardScaler()
        self.target_model = xgb.XGBRegressor(n_estimators=args.target_model.n_estimators)

        self.target = target


    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        pseudo_out = data_dict[f'pseudo_{self.target}_out']
        weights = data_dict[f'pseudo_{self.target}_weights']

        return cov_f, pseudo_out, weights

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        return cov_f, out_pot0, out_pot1, mu0, mu1

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, pseudo_out, weights = self.prepare_train_data(train_data_dict)

        self.target_model.fit(cov_f, pseudo_out, sample_weight=weights)

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str, target: str):
        assert self.target == target

        cov_f, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        pseudo_out_pred = self.target_model.predict(cov_f).reshape(-1, 1)

        rpehe = np.sqrt(((pseudo_out_pred - (out_pot1 - out_pot0)) ** 2).mean())
        rpehe_oracle = np.sqrt((((mu1 - mu0) - (out_pot1 - out_pot0)) ** 2).mean()) if self.oracle_available else None

        results = {f'{prefix}_{target}_rpehe': rpehe.item(), f'{prefix}_{target}_rpehe_oracle': rpehe_oracle.item()}

        if log:
            self.mlflow_logger.log_metrics(results, step=0)
        return results

    def evaluate_pot_mse(self, data_dict: dict, log: bool, prefix: str, target: str):
        assert self.target == target

        cov_f, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)
        mu = mu0 if target in ['mu0', 'y0'] else mu1
        out_pot = out_pot0 if target in ['mu0', 'y0'] else out_pot1

        pseudo_out_pred = self.target_model.predict(cov_f).reshape(-1, 1)

        mse = np.sqrt((((pseudo_out_pred - out_pot)) ** 2).mean())
        mse_oracle = np.sqrt((((mu - out_pot)) ** 2).mean()) if self.oracle_available else None

        results = {f'{prefix}_mse_{target}': mse.item(), f'{prefix}_mse_{target}_oracle': mse_oracle.item()}
        if log:
            self.mlflow_logger.log_metrics(results, step=0)
        return results

