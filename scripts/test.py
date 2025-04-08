#!/usr/bin/env python3
import lightning as L
import os
import argparse
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from bio2token.data.collate_fn import PadAndStack, PadAndStackConfig
from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.data.dataset import DatasetModule, DatasetConfig
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from bio2token.utils.lightning import NetworkModule, find_lowest_val_loss_checkpoint
from bio2token.utils.callbacks import StorerCallback

DEBUG = False
if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    """
    Main function to test an .

    This function performs the following steps:
    1. Parse command-line arguments to get the configuration file path.
    2. Load the configuration from a YAML file.
    3. Instantiate the model using the configuration.
    4. Load the model checkpoint based on the configuration.
    5. Set up the data module for testing.
    6. Instantiate the LightningModule for the model.
    7. Add necessary callbacks for inference.
    8. Instantiate the trainer and test the model.

    Command-line Arguments:
    --config: str
        Path to the YAML configuration file (default: "test.yaml").
        Config file must be in the configs/ folder.
    """
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test.yaml")
    args = parser.parse_args()

    # STEP 1: Load config yaml file
    global_configs = utilsyaml_to_dict(args.config)

    # Add MLflow setup after loading config
    tracking_uri = (
        f"http://{global_configs['mlflow']['tracking_server_host']}:{global_configs['mlflow']['tracking_server_port']}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.enable_system_metrics_logging()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # STEP 2: Instantiate model.
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)

    # Load checkpoint
    ckpt_path = f"{global_configs['infer']['checkpoint_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}/last.ckpt"
    if global_configs["infer"].get("checkpoint_type") == "best":
        ckpt_path = find_lowest_val_loss_checkpoint(
            checkpoint_dir=f"{global_configs['infer']['checkpoint_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}",
            checkpoint_monitor=global_configs["infer"]["checkpoint_monitor"],
            checkpoint_mode=global_configs["infer"]["checkpoint_mode"],
        )
    ckpt_path_name = ckpt_path.split("/")[-1].strip(".ckpt")
    # STEP 4: Instantiate our datamodule and collate function
    data_config = pi_instantiate(DatasetConfig, yaml_dict=global_configs["data"])
    dm = DatasetModule(config=data_config)
    collate_fn_config = pi_instantiate(PadAndStackConfig, yaml_dict=global_configs["collate_fn"])
    collate_fn = PadAndStack(collate_fn_config)
    dm.set_collate_fn(collate_fn)
    dm.setup(stage="test")

    # STEP 5: Instantiate the LightningModule of the model.
    sma = NetworkModule(model=model)

    # STEP 6: Add our callbacks. We no longer automatically add any callbacks, so any that you want need to be explicitly added.
    callbacks = [
        StorerCallback(
            save_dir=f"{global_configs['infer']['results_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}/{ckpt_path_name}/{global_configs['data']['ds_name']}/{global_configs['data']['dataset'][global_configs['data']['ds_name']]['test_split']}/",
            try_keeping_save=global_configs["infer"]["keys_to_save"],
            try_adding_summary=global_configs["infer"]["keys_to_summarize"],
        )
    ]

    # STEP 7: Instantiate our trainer
    runs = mlflow.search_runs()
    if global_configs["infer"]["run_id"] not in runs["run_id"]:
        run_id = None
    else:
        run_id = global_configs["infer"]["run_id"]
    logger = MLFlowLogger(
        experiment_name=global_configs["infer"]["experiment_name"],
        tracking_uri=tracking_uri,
        run_id=run_id,
    )
    trainer = pi_instantiate(L.Trainer, yaml_dict=global_configs["lightning_trainer"], callbacks=callbacks, logger=logger)

    # STEP 8: Test our model within MLflow run context
    # check if run_id is already in mlflow
    with mlflow.start_run(run_id=run_id):  # Use the same run_id as the trained model
        trainer.test(model=sma, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
