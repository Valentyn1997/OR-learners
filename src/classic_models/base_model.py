import ray
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from omegaconf import DictConfig
from ray import tune
import logging
from pytorch_lightning.loggers import MLFlowLogger

from src.models.utils import fit_eval_kfold

logger = logging.getLogger(__name__)


class BaseModel:

    val_metric = None
    name = None

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):

        self.dim_cov = args.dataset.dim_cov
        self.cov_scaler = StandardScaler()
        self.treat_options = [0.0, 1.0]
        self.oracle_available = args.dataset.oracle_available

        # Model hyparams
        self.hparams = args

        # MlFlow Logger
        self.mlflow_logger = mlflow_logger

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            assert k in model_args.keys()
            model_args[k] = new_model_args[k]

    def prepare_train_data(self, data_dict: dict):
        raise NotImplementedError()

    def prepare_eval_data(self, data_dict: dict):
        raise NotImplementedError()

    def fit(self, train_data_dict: dict, log: bool):
        raise NotImplementedError()

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        raise NotImplementedError()

    def get_outcomes(self, data_dict):
        raise NotImplementedError()

    def finetune(self, train_data_dict: dict, resources_per_trial: dict, val_data_dict: dict = None):
        """
        Hyperparameter tuning with ray[tune]
        """

        logger.info(f"Running hyperparameters selection with {self.hparams[self.name]['tune_range']} trials")
        logger.info(f'Using {self.val_metric} for hyperparameters selection')
        ray.init(num_cpus=3)

        hparams_grid = {k: getattr(tune, self.hparams[self.name]['tune_type'])(list(v))
                        for k, v in self.hparams[self.name]['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(fit_eval_kfold,
                                                 model_cls=self.__class__,
                                                 train_data_dict=deepcopy(train_data_dict),
                                                 val_data_dict=deepcopy(val_data_dict),
                                                 name=self.name, subnet_name=None,
                                                 kind=self.kind if hasattr(self, 'kind') else None,
                                                 orig_hparams=self.hparams),
                            resources_per_trial=resources_per_trial,
                            raise_on_failed_trial=False,
                            metric='val_metric',
                            mode="min",
                            config=hparams_grid,
                            num_samples=self.hparams[self.name]['tune_range'],
                            name=f"{self.__class__.__name__}",
                            max_failures=1,
                            )
        ray.shutdown()

        logger.info(f"Best hyperparameters found for {self.name}: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_hparams(self.hparams[self.name], analysis.best_config)
        self.__init__(self.hparams, mlflow_logger=self.mlflow_logger, kind=self.kind if hasattr(self, 'kind') else None)
        return self
