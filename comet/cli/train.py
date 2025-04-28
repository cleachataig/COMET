#!/usr/bin/env python3

# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

Command for training new Metrics.
=================================

e.g:
```
    comet-train --cfg configs/models/regression_metric.yaml --seed_everything 12
```

For more details run the following command:
```
    comet-train --help
```
"""

import os
import copy
import json
from pprint import pprint
import logging
import warnings

import torch
mem_limit = os.getenv("MEMORY_LIMIT")
if mem_limit is not None:
    torch.cuda.set_per_process_memory_fraction(float(mem_limit), device=0)
    print(f"Limiting memory to {mem_limit} of GPU 0")

from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.trainer.trainer import Trainer

import optuna
from optuna.samplers import GridSampler
import optuna.visualization as vis

from comet.models import (RankingMetric, ReferencelessRegression,
                          RegressionMetric, UnifiedMetric)

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
progress_bar_callback = TQDMProgressBar(refresh_rate=10000)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training COMET models.")
    parser.add_argument(
        "--seed_everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(
        ReferencelessRegression, "referenceless_regression_metric"
    )
    parser.add_subclass_arguments(RankingMetric, "ranking_metric")
    parser.add_subclass_arguments(UnifiedMetric, "unified_metric")
    parser.add_subclass_arguments(EarlyStopping, "early_stopping")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(Trainer, "trainer")
    parser.add_argument(
        "--load_from_checkpoint",
        help="Loads a model checkpoint for fine-tuning",
        default=None,
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Strictly enforce that the keys in checkpoint_path match the keys returned by this module's state dict.",
    )
    parser.add_argument(
    "--search",
    action="store_true",
    help="Enable hyperparameter search with Optuna.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        default=None,
        help="Path to JSON file defining the hyperparameter search space.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=16,
        help="Number of trials for parameter search.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of concurrent jobs.",
    )
    return parser


def initialize_trainer(configs) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(configs.model_checkpoint.init_args)
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(configs.early_stopping.init_args)
    )
    trainer_args = namespace_to_dict(configs.trainer.init_args)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback, lr_monitor, progress_bar_callback]
    print("TRAINER ARGUMENTS: ")
    pprint(trainer_args)
    #print(json.dumps(trainer_args, indent=4, default=lambda x: x.__dict__))
    trainer = Trainer(**trainer_args)
    return trainer


def initialize_model(configs):
    print("MODEL ARGUMENTS: ")
    if configs.regression_metric is not None:
        print(
            json.dumps(
                configs.regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = RegressionMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.regression_metric.init_args),
            )
        else:
            model = RegressionMetric(
                **namespace_to_dict(configs.regression_metric.init_args)
            )
    elif configs.referenceless_regression_metric is not None:
        print(
            json.dumps(
                configs.referenceless_regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = ReferencelessRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.referenceless_regression_metric.init_args),
            )
        else:
            model = ReferencelessRegression(
                **namespace_to_dict(configs.referenceless_regression_metric.init_args)
            )
    elif configs.ranking_metric is not None:
        print(
            json.dumps(
                configs.ranking_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = RankingMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.ranking_metric.init_args),
            )
        else:
            model = RankingMetric(**namespace_to_dict(configs.ranking_metric.init_args))
    elif configs.unified_metric is not None:
        print(
            json.dumps(
                configs.unified_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = UnifiedMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.unified_metric.init_args),
            )
        else:
            model = UnifiedMetric(**namespace_to_dict(configs.unified_metric.init_args))
    else:
        raise Exception("Model configurations missing!")

    return model

def save_visualizations(study, output_dir="optuna"):
    os.makedirs(output_dir, exist_ok=True)
    vis.plot_optimization_history(study).write_html(f"{output_dir}/history.html")
    vis.plot_param_importances(study).write_html(f"{output_dir}/importances.html")
    vis.plot_parallel_coordinate(study).write_html(f"{output_dir}/parallel.html")
    vis.plot_contour(study).write_html(f"{output_dir}/contour.html")
    print(f"Saved visualizations to {output_dir}")


def hyperparameter_search(cfg):
    assert cfg.search_space is not None, "You must specify --search_space."

    with open(cfg.search_space) as f:
        search_space = json.load(f)
    if cfg.regression_metric is not None:
        model_type = "regression"
        pretrained = cfg.regression_metric.init_args.pretrained_model
    elif cfg.ranking_metric is not None:
        model_type = "ranking"
        pretrained = cfg.ranking_metric.init_args.pretrained_model
    else:
        model_type = "unknown"
        pretrained = "unknown"

    study_name=f"{model_type}-{pretrained.replace('/', '_')}"
    os.makedirs("optuna/", exist_ok=True)
    storage_url = f"sqlite:///optuna/{study_name}.db"
    old_storage_url = f"sqlite:///optuna/old/{study_name}.db"
    old_study = optuna.load_study(study_name=study_name, storage=old_storage_url)
    existing_trials = {}
    for trial in old_study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Ensure that "keep_embeddings_frozen" is set to True for old trials
            trial_params = trial.params.copy()
            if "keep_embeddings_frozen" not in trial_params:
                trial_params["keep_embeddings_frozen"] = True  # Assume True for old trials
            # Store trial parameters and their corresponding value
            existing_trials[frozenset(trial_params.items())] = trial.value
    print(existing_trials)
    
    def objective(trial):
        
        trial_params = {}
        for param, values in search_space.items():
            value = trial.suggest_categorical(param, values)
            trial_params[param] = value  # Store the suggested value for later use
        
        # Check if the trial parameters already exist in the old study
        trial_params_frozen = frozenset(trial_params.items())
        if trial_params_frozen in existing_trials:
            old_trial_value = existing_trials[trial_params_frozen]
            print(f"[Trial {trial.number}] Skipping trial with parameters {trial_params} as it already exists in the old study.")
            print(f"Old trial value: {old_trial_value}")
            raise optuna.TrialPruned()  # Prune the trial since it's already completed in the previous study

        cfg_trial = copy.deepcopy(cfg) 
        trial_id = trial.number
        base_log_dir = cfg_trial.trainer.init_args.default_root_dir
        # If it's wrapped in a config object (e.g., OmegaConf), resolve the string
        if hasattr(base_log_dir, "__str__"):
            base_log_dir = str(base_log_dir)

        log_dir = os.path.join(base_log_dir, f"trial_{trial_id}")
        cfg_trial.trainer.init_args.default_root_dir = log_dir
        
        try:
            # Set the parameters in the configuration
            if cfg_trial.regression_metric is not None:
                for param, value in trial_params.items():
                    setattr(cfg_trial.regression_metric.init_args, param, value)
            elif cfg_trial.ranking_metric is not None:
                for param, value in trial_params.items():
                    setattr(cfg_trial.ranking_metric.init_args, param, value)

            # Initialize the model and trainer with the updated config
            model = initialize_model(cfg_trial)
            trainer = initialize_trainer(cfg_trial)
            trainer.fit(model)

            # Adjust based on your validation metric name
            val_metric = trainer.callback_metrics.get("val_kendall")
            if val_metric is not None:
                return val_metric.item()
            return float("inf")  # fallback
        
        except Exception as e:
            print(f"[Trial {trial.number}] Failed with error: {e}")
            raise optuna.TrialPruned()

    study = optuna.create_study(
        direction="minimize",
        sampler=GridSampler(search_space),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)
    save_visualizations(study, output_dir=f"optuna/{study_name}")

    print("Best hyperparameters found:")
    print(study.best_params)

def train_command() -> None:
    parser = read_arguments()
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    if cfg.search:
        hyperparameter_search(cfg)
    else:
        trainer = initialize_trainer(cfg)
        model = initialize_model(cfg)

        # Related to train/val_dataloaders:
        # 2 workers per gpu is enough! If set to the number of cpus on this machine
        # it throws another exception saying its too many workers.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)


if __name__ == "__main__":
    train_command()
