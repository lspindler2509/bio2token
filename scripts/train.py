#!/usr/bin/env python3
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import mlflow
import os
import argparse
from bio2token.utils.callbacks import CustomProgressBar, StopOnNaNCallback
from bio2token.utils.lightning import OptimizerConfig, NetworkModule, find_lowest_val_loss_checkpoint
from bio2token.data.collate_fn import PadAndStack, PadAndStackConfig
from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.data.dataset import DatasetModule, DatasetConfig
from bio2token.utils.mlflow import flatten_config
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate

DEBUG = False
if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    """
    Main function to train an autoencoder.

    This function performs the following steps:
    1. Parses command-line arguments to obtain the configuration file path.
    2. Loads the configuration from a YAML file.
    3. Instantiates the model and data module using the configurations.
    4. Sets up the optimizer and network module.
    5. Adds necessary callbacks for training.
    6. Configures the logger for monitoring training progress.
    7. Instantiates the PyTorch Lightning trainer.
    8. Initiates the training process, optionally continuing from a checkpoint.

    Command-line Arguments:
    --config: str
        Path to the YAML configuration file (default: "train.yaml").
        Config file must be in the configs/ folder.
    """
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    args = parser.parse_args()

    # STEP 1: Load config yaml file
    global_configs = utilsyaml_to_dict(args.config)

    # STEP 2: Start mlflow
    tracking_uri = (
        f"http://{global_configs['mlflow']['tracking_server_host']}:{global_configs['mlflow']['tracking_server_port']}"
    )
    experiment_name = global_configs["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.enable_system_metrics_logging()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # STEP 3: Instantiate model.
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)

    # STEP 4: Instantiate our datamodule and collate function
    data_config = pi_instantiate(DatasetConfig, yaml_dict=global_configs["data"])
    dm = DatasetModule(config=data_config)
    collate_fn_config = pi_instantiate(PadAndStackConfig, yaml_dict=global_configs["collate_fn"])
    collate_fn = PadAndStack(collate_fn_config)
    dm.set_collate_fn(collate_fn)

    # STEP 5: Instantiate the LightningModule of the model.
    optimizer_config = pi_instantiate(OptimizerConfig, yaml_dict=global_configs["optimizer"])
    sma = NetworkModule(config=optimizer_config, model=model)

    # STEP 6: Get our version and logger.
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    print("{epoch:04d}" + "-{" + optimizer_config.checkpointing.checkpoint_monitor + ":.2f}-best-checkpoint")
    # STEP 7: Add our callbacks. We no longer automatically add any callbacks, so any that you want need to be explicitly added.
    callbacks = [
        CustomProgressBar(),
        StopOnNaNCallback(),
        ModelCheckpoint(
            dirpath=f"{optimizer_config.checkpointing.checkpoint_dir}/{experiment_name}/{logger.run_id}",  # Directory to save checkpoints
            filename="{epoch:04d}" + "-{" + optimizer_config.checkpointing.checkpoint_monitor + ":.2f}-best-checkpoint",
            save_top_k=optimizer_config.checkpointing.checkpoint_k_best_to_save,  # Save only the best model
            monitor=optimizer_config.checkpointing.checkpoint_monitor,  # Monitor a specific metric
            mode=optimizer_config.checkpointing.checkpoint_mode,  # 'min' for loss, 'max' for accuracy-based metrics
            save_last=True,
        ),
    ]

    # STEP 8: Instantiate our trainer.
    trainer = pi_instantiate(
        L.Trainer,
        yaml_dict=global_configs["lightning_trainer"],
        callbacks=callbacks,
        logger=logger,
    )

    # STEP 9: Train our model, with continue training if requested.
    # if we want to start from a specific checkpoint instead of from scratch, this will grab it from mlflow - need to specify run_id in the config for now.
    if optimizer_config.continue_training and optimizer_config.pretrained_model.run_id is not None:
        ckpt_path = f"{optimizer_config.pretrained_model.checkpoint_dir}/{experiment_name}/{optimizer_config.pretrained_model.run_id}/last.ckpt"
        if optimizer_config.pretrained_model.checkpoint_type == "best":
            ckpt_path = find_lowest_val_loss_checkpoint(
                checkpoint_dir=f"{optimizer_config.pretrained_model.checkpoint_dir}/{experiment_name}/{optimizer_config.pretrained_model.run_id}",
                checkpoint_monitor=optimizer_config.pretrained_model.checkpoint_monitor,
                checkpoint_mode=optimizer_config.pretrained_model.checkpoint_mode,
            )
    else:
        ckpt_path = None

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_params(flatten_config(global_configs))
        trainer.fit(model=sma, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
