import logging
import hydra
import torch
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit

from src.models.utils import subset_by_indices
from src import ROOT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name=f'config_classic.yaml', config_path='../config/')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of dataset
    torch.set_default_device(args.exp.device)
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)
    data_dicts = dataset.get_data()
    if not args.dataset.collection:
        data_dicts = [data_dicts]
    if args.dataset.dataset_ix is not None:
        data_dicts = [data_dicts[args.dataset.dataset_ix]]
        specific_ix = True
    else:
        specific_ix = False

    for ix, data_dict in enumerate(data_dicts):

        # ================= Train / test split =================
        args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix
        if args.dataset.train_test_splitted:
            data_dicts = [data_dict]
        else:
            ss = ShuffleSplit(n_splits=args.dataset.n_shuffle_splits, random_state=2 * args.exp.seed, test_size=args.dataset.test_size)
            indices = list(ss.split(data_dict['cov_f']))
            data_dicts = [(subset_by_indices(data_dict, train_index), subset_by_indices(data_dict, test_index)) for (train_index, test_index) in indices]

        # ================= Train-validation split for downstream models =================
        for i, (train_data_dict, test_data_dict) in enumerate(data_dicts):

            # ================= Mlflow init =================
            experiment_name = f'xgboost/{args.dataset.name}'
            mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                         tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

            # ss = ShuffleSplit(n_splits=1, random_state=args.exp.seed, test_size=args.dataset.test_size)
            # ttrain_index, val_index = list(ss.split(train_data_dict['cov_f']))[0]
            #
            # ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, ttrain_index), \
            #     subset_by_indices(train_data_dict, val_index)

            # ================= Fitting propensity model =================
            prop_model_cov = instantiate(args.prop_model_cov, args, mlflow_logger, _recursive_=True)

            # Finetuning for the first split
            if args.prop_model_cov.tune_hparams and i == 0:
                prop_model_cov.finetune(train_data_dict, {'cpu': 0.5})

            # Training
            logger.info(f'Fitting propensity model for sub-dataset {args.dataset.dataset_ix}.')
            prop_model_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

            # Evaluation
            results_out_cov = prop_model_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
            logger.info(f'Out-sample performance propensity cov: {results_out_cov}')

            # Getting propensity weights
            for data_dict in [train_data_dict, test_data_dict]:
                data_dict['prop_pred_cov'] = prop_model_cov.get_prop_predictions(data_dict=data_dict)


            # ================= Fitting mu model =================
            mu_model_cov = instantiate(args.mu_model_cov, args, mlflow_logger, _recursive_=True)

            # Finetuning for the first split
            if args.mu_model_cov.tune_hparams and i == 0:
                mu_model_cov.finetune(train_data_dict, {'cpu': 0.5})

            # Training
            logger.info(f'Fitting mu-model for sub-dataset {args.dataset.dataset_ix}.')
            mu_model_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

            # Evaluation
            results_out_cov = mu_model_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
            logger.info(f'Out-sample performance mu-model: {results_out_cov}')

            # Getting propensity weights
            for data_dict in [train_data_dict, test_data_dict]:
                data_dict['mu_pred0_cov'], data_dict['mu_pred1_cov'] = mu_model_cov.get_outcomes(data_dict=data_dict)


            # ================= Fitting target nets =================
            for target in args.exp.targets:

                # ================= Fitting a target model =================
                logger.info(f'Fitting {target}-target net for sub-dataset {args.dataset.dataset_ix}.')
                target_model = instantiate(args.target_model, args, mlflow_logger, target, _recursive_=True)

                # ================= Generating pseudo-outcomes =================
                train_data_dict[f'pseudo_{target}_out'], train_data_dict[f'pseudo_{target}_weights'] = mu_model_cov.get_pseudo_out(data_dict=train_data_dict, target=target)

                target_model.fit(train_data_dict=train_data_dict, log=args.exp.logging)

                if target in ['mu0', 'mu1', 'y0', 'y1']:
                    results_in = target_model.evaluate_pot_mse(data_dict=train_data_dict, log=args.exp.logging, prefix='in_target', target=target)
                    results_out = target_model.evaluate_pot_mse(data_dict=test_data_dict, log=args.exp.logging, prefix='out_target', target=target)

                elif target in ['cate', 'rcate', 'ivw_pi_cate', 'ivw_a_cate']:
                    results_in = target_model.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in_target', target=target)
                    results_out = target_model.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out_target', target=target)
                else:
                    raise NotImplementedError()

                logger.info(f'MSE In-sample performance: {results_in}. MSE Out-sample performance: {results_out}')


            mlflow_logger.log_hyperparams(args) if args.exp.logging else None
            mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return results_in, results_out


if __name__ == "__main__":
    main()
