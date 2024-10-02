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


@hydra.main(config_name=f'config.yaml', config_path='../config/')
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
            experiment_name = f'{args.repr_net.name}/{args.dataset.name}/{args.repr_net.repr_net_type}'
            mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                         tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

            # ss = ShuffleSplit(n_splits=1, random_state=args.exp.seed, test_size=args.dataset.test_size)
            # ttrain_index, val_index = list(ss.split(train_data_dict['cov_f']))[0]
            #
            # ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, ttrain_index), \
            #     subset_by_indices(train_data_dict, val_index)

            # ================= Fitting representation net =================

            if args.repr_net.has_prop_net_cov:  # For BWCFR
                prop_net_cov = instantiate(args.prop_net_cov, args, mlflow_logger, 'cov', _recursive_=True)

                # Finetuning for the first split
                if args.prop_net_cov.tune_hparams and i == 0:
                    prop_net_cov.finetune(train_data_dict, {'cpu': 0.1, 'gpu': 0.24})

                # Training
                logger.info(f'Fitting propensity net for sub-dataset {args.dataset.dataset_ix}.')
                prop_net_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

                # Evaluation
                results_out_cov = prop_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
                logger.info(f'Out-sample performance propensity cov: {results_out_cov}')

                # Getting propensity weights
                for data_dict in [train_data_dict, test_data_dict]:
                    data_dict['prop_pred_cov'] = prop_net_cov.get_prop_predictions(data_dict=data_dict)

            repr_net = instantiate(args.repr_net, args, mlflow_logger, _recursive_=True)

            # Finetuning for the first split
            if args.repr_net.tune_hparams and i == 0:
                repr_net.finetune(train_data_dict, {'cpu': 0.1, 'gpu': 0.24})

            # Training
            logger.info(f'Fitting a representation net for sub-dataset {args.dataset.dataset_ix}.')
            repr_net.fit(train_data_dict=train_data_dict, log=args.exp.logging)

            results_in = repr_net.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in_repr')
            results_out = repr_net.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out_repr')
            logger.info(f'PEHE In-sample performance: {results_in}. PEHE Out-sample performance: {results_out}')

            results_in = repr_net.evaluate_pot_mses(data_dict=train_data_dict, log=args.exp.logging, prefix='in_repr')
            results_out = repr_net.evaluate_pot_mses(data_dict=test_data_dict, log=args.exp.logging, prefix='out_repr')
            logger.info(f'MSE In-sample performance: {results_in}. MSE Out-sample performance: {results_out}')

            # Inferring representations & predictions
            for data_dict in [train_data_dict, test_data_dict]:
                data_dict['repr_f'] = repr_net.get_representations(data_dict=data_dict)
                data_dict['mu_pred0_repr'], data_dict['mu_pred1_repr'] = repr_net.get_outcomes(data_dict=data_dict)

                if args.repr_net.has_prop_net_repr:
                    data_dict['prop_pred_repr'] = repr_net.prop_net_repr.get_prop_predictions(data_dict=data_dict)

            if args.exp.logging and args.exp.save_results:
                save_dir = f'{ROOT_PATH}/mlruns/{mlflow_logger.experiment_id}/{mlflow_logger.run_id}/artifacts'
                np.save(f'{save_dir}/train_data_dict.npy', train_data_dict)
                np.save(f'{save_dir}/test_data_dict.npy', test_data_dict)
                torch.save(repr_net, f'{save_dir}/repr_net.pt')

            # ================= Fitting nuisance nets =================
            if args.repr_net.repr_net_type not in ['res_flow', 'cnf'] and args.exp.targets is not None and len(args.exp.targets) > 0:
                if not args.repr_net.has_prop_net_cov:
                    prop_net_cov = instantiate(args.prop_net_cov, args, mlflow_logger, 'cov', _recursive_=True)

                    # Finetuning for the first split
                    if args.prop_net_cov.tune_hparams and i == 0:
                        prop_net_cov.finetune(train_data_dict, {'cpu': 0.1, 'gpu': 0.24})

                    # Training
                    logger.info(f'Fitting propensity net for sub-dataset {args.dataset.dataset_ix}.')
                    prop_net_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

                    # Evaluation
                    results_out_cov = prop_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
                    logger.info(f'Out-sample performance propensity cov: {results_out_cov}')

                    # Getting propensity weights
                    for data_dict in [train_data_dict, test_data_dict]:
                        data_dict['prop_pred_cov'] = prop_net_cov.get_prop_predictions(data_dict=data_dict)

                if not args.mu_net_cov.use_repr_pred:
                    mu_net_cov = instantiate(args.mu_net_cov, args, mlflow_logger, _recursive_=True)

                    # Finetuning for the first split
                    if args.mu_net_cov.tune_hparams and i == 0:
                        mu_net_cov.finetune(train_data_dict, {'cpu': 0.1, 'gpu': 0.24})

                    # Training
                    logger.info(f'Fitting mu-net for sub-dataset {args.dataset.dataset_ix}.')
                    mu_net_cov.fit(train_data_dict=train_data_dict, log=args.exp.logging)

                    # Evaluation
                    results_out_cov = mu_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
                    logger.info(f'Out-sample performance mu-net: {results_out_cov}')

                    # Getting propensity weights
                    for data_dict in [train_data_dict, test_data_dict]:
                        data_dict['mu_pred0_cov'], data_dict['mu_pred1_cov'] = mu_net_cov.get_outcomes(data_dict=data_dict)

                else:
                    for data_dict in [train_data_dict, test_data_dict]:
                        data_dict['mu_pred0_cov'], data_dict['mu_pred1_cov'] = repr_net.get_outcomes(data_dict=data_dict)


            # ================= Fitting target nets =================
            for target in args.exp.targets:
                if target in ['mu0', 'mu1', 'y0', 'y1'] and args.repr_net.repr_net_type in ['res_flow', 'cnf']: #  IPTW-R-flow is already orthogonal (e.g., BWCFR-flow and CFRISW-flow)
                    continue
                else:
                    # ================= Generating pseudo-outcomes =================
                    if target in ['mu0', 'mu1', 'cate']:
                        train_data_dict[f'pseudo_{target}'] = repr_net.get_pseudo_out(data_dict=train_data_dict, target=target)

                    # ================= Fitting a target net =================
                    logger.info(f'Fitting {target}-target net for sub-dataset {args.dataset.dataset_ix}.')
                    target_net = instantiate(args.target_net, args, mlflow_logger, target, _recursive_=True)
                    target_net.fit(train_data_dict=train_data_dict, log=args.exp.logging)

                    if target in ['mu0', 'mu1', 'y0', 'y1']:
                        results_in = target_net.evaluate_pot_mse(data_dict=train_data_dict, log=args.exp.logging, prefix='in_target', target=target)
                        results_out = target_net.evaluate_pot_mse(data_dict=test_data_dict, log=args.exp.logging, prefix='out_target', target=target)

                    elif target in ['cate', 'rcate']:
                        results_in = target_net.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in_target', target=target)
                        results_out = target_net.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out_target', target=target)
                    else:
                        raise NotImplementedError()

                    logger.info(f'MSE In-sample performance: {results_in}. MSE Out-sample performance: {results_out}')


            mlflow_logger.log_hyperparams(args) if args.exp.logging else None
            mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return results_in, results_out


if __name__ == "__main__":
    main()
