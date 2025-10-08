Orthogonal representation learning for CATE estimation  
==============================

A novel class of Neyman-orthogonal learners for causal quantities defined at the representation level

![image](https://github.com/user-attachments/assets/c5d6da01-b2b5-412f-9bbb-6d3c74a0e33a)

The project is built with the following Python libraries:
1. [PyTorch](https://pytorch.org/)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking
4. [normflows](https://github.com/VincentStimper/normalizing-flows) - a PyTorch package for normalizing flows


## Setup

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000 --gunicorn-opts "--timeout 280"`

To access the MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.

### Semi-synthetic datasets setup

Before running semi-synthetic experiments, place datasets in the corresponding folders:
- [IHDP100 dataset](https://www.fredjo.com/): ihdp_npci_1-100.test.npz and ihdp_npci_1-100.train.npz to `data/ihdp100/`
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition): to `data/acic2016/`
```
 ── data/acic_2016
    ├── synth_outcomes
    |   ├── zymu_<id0>.csv   
    |   ├── ... 
    │   └── zymu_<id14>.csv 
    ├── ids.csv
    └── x.csv 
```

## Experiments

The main training script is universal for different methods and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `config/` folder.

Generic script with logging and fixed random seed is the following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +repr_net=<model> exp.seed=10
```

### Datasets
One needs to specify a dataset / dataset generator (and some additional parameters, e.g. train size for the synthetic data `dataset.n_samples_train=1000`, or a subset index for ACIC 2016 data `dataset.dataset_ix=0`):
- Synthetic data (adapted from https://arxiv.org/abs/1810.02894): `+dataset=synthetic`
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp100` 
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition) dataset: `+dataset=acic2016`

### Models
#### Stage 0.
One needs to choose a model and then fill in the specific hyperparameters (they are left blank in the configs):
- [TARNet/TARFlow](https://arxiv.org/abs/1606.03976): `+repr_net=tarnet +repr_net_type=dense` / `+repr_net=tarnet +repr_net_type=res_flow`
- [BNN/BNNFlow](https://arxiv.org/abs/1605.03661): `+repr_net=bnnet +repr_net_type=dense` / `+repr_net=bnnet +repr_net_type=res_flow`
- [CFR/CFRFlow](https://arxiv.org/abs/1606.03976): `+repr_net=cfrnet +repr_net_type=dense` / `+repr_net=cfrnet +repr_net_type=res_flow`
- [RCFR/RCFRFlow](https://arxiv.org/abs/2001.07426): `+repr_net=rcfrnet +repr_net_type=dense` / `+repr_net=rcfrnet +repr_net_type=res_flow`
- [CFR-ISW/CFRNet-ISW](https://www.ijcai.org/proceedings/2019/0815.pdf): `+repr_net=cfrisw +repr_net_type=dense` / `+repr_net=cfrisw +repr_net_type=res_flow`
- [BWCFR/BWCFRFlow](https://arxiv.org/abs/2010.12618): `+repr_net=bwcfr +repr_net_type=dense` / `+repr_net=bwcfr +repr_net_type=res_flow`

Models already have the best hyperparameters saved, for each model - dataset and different sizes of the representation. One can access them via: `+repr_net/<dataset>_hparams/<model>=<n_samples_train>` or `+model/<dataset>_hparams/<model>/<n_samples_train>=<ipm_params>` etc. To perform manual hyperparameter tuning use the flags `repr_net.tune_hparams=True`, and then see `repr_net.hparams_grid`. 

#### Stage 1.
Stage 1 models are propensity networks (src/models/prop_nets.py) and outcome networks (src/models/mu_nets.py). The hyperparameters were tuned together with the stage 0 models and are stored in the same YAML files. To perform manual hyperparameter tuning use the flags `prop_net_cov.tune_hparams=True` and `mu_net_cov.tune_hparams=True`.

#### Stage 2.
Stage 2 models are defined in config/config.yaml and src/models/target_net.py. One needs to specify the list of second-stage models to fit for `exp.targets`:
- CAPOs estimation with $\text{DR}^{\text{K}}_a$-learner: `exp.targets="['mu0', 'mu1']"`
- CAPOs estimation with $\text{DR}^{\text{FS}}_a$-learner: `exp.targets="['y0', 'y1']"`
- CATE estimation with $\text{DR}^{\text{K}}$: `exp.targets="['cate']"`
- CATE estimation with $\text{R}$: `exp.targets="['rcate']"`
- CATE estimation with $\text{IVW}$: `exp.targets="['ivw_pi_cate']"`


### Examples
Example of running TARFlow (CFR w/ $\alpha = 0$) without tuning based on synthetic data with n_train = 500 (stage 0):
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/train.py -m +dataset=synthetic +repr_net=tarnet +repr_net/synthetic_hparams/tarnet_res_flow=\'500\' exp.logging=True exp.device=cuda exp.seed=10 exp.targets="[]"
```

Example of running BWCFRFlow without tuning based on synthetic data with n_train = 500, $\alpha \in \\{0.01, 0.02,	0.05\\}$, and IPM = MMD (stages 0-2):
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/train.py -m +dataset=synthetic +repr_net=bwcfr +repr_net/synthetic_hparams/bwcfr_res_flow=\'500\' exp.logging=True repr_net.ipm=mmd repr_net.alpha=0.01,0.02,0.05 exp.device=cuda exp.seed=10 exp.targets="['mu0', 'mu1', 'y0', 'y1', 'cate', 'rcate']"
```

Example of all-stages tuning of CFR based on the 0-th subset of IHDP100 dataset with IPM = WM and $\alpha = 0.1$ (stages 0-2):
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. -m +dataset=ihdp100 +repr_net=cfrnet exp.logging=True exp.device=cuda dataset.dataset_ix=0 repr_net.ipm=wass repr_net.alpha=0.1 repr_net.tune_hparams=True prop_net_cov.tune_hparams=True mu_net_cov.tune_hparams=True exp.targets="['mu0', 'mu1', 'y0', 'y1', 'cate', 'rcate']"
```

-------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
