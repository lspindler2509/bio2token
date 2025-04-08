#!/usr/bin/env python3
import os
import argparse
from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from bio2token.utils.lightning import find_lowest_val_loss_checkpoint
from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe, write_pdb
from bio2token.data.utils.molecule_conventions import ABBRS
import torch
import json

DEBUG = False
if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_pdb.yaml")
    args = parser.parse_args()

    # STEP 1: Load config yaml file
    global_configs = utilsyaml_to_dict(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdb_path = global_configs["data"]["pdb_path"]
    dict = pdb_2_dict(
        pdb_path,
        chains=global_configs["data"]["chains"] if "chains" in global_configs["data"] else None,
    )
    structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
        dict["seq"],
        dict["res_types"],
        dict["coords_groundtruth"],
        dict["atom_names"],
        dict["res_atom_start"],
        dict["res_atom_end"],
    )
    batch = {
        "structure": torch.tensor(structure).float(),
        "unknown_structure": torch.tensor(unknown_structure).bool(),
        "residue_ids": torch.tensor(residue_ids).long(),
        "token_class": torch.tensor(token_class).long(),
    }
    batch = {k: v[~batch["unknown_structure"]] for k, v in batch.items()}
    batch = compute_masks(batch, structure_track=True)
    batch = {k: v[None].to(device) for k, v in batch.items()}

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
    state_dict = torch.load(ckpt_path)["state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    model.eval()
    model.to(device)

    # STEP 3: Inference
    batch = model(batch)
    for k, v in batch["losses"].items():
        batch[k] = v.item()

    # STEP 4: Print and save results
    for key in global_configs["infer"]["keys_to_summarize"]:
        if key in batch:
            print(f"{key}: {batch[key]}")

    gt_coords = batch["structure"][0].detach().cpu().numpy()
    rec_coords = batch["all_atom_coords"][0].detach().cpu().numpy()
    atom_types = atom_names_reordered
    residue_types = [ABBRS[res.split("_")[0]][res.split("_")[1]] for res in residue_name]
    if "chains" in global_configs["data"] and global_configs["data"]["chains"] is not None:
        chains = "_".join(global_configs["data"]["chains"])
    else:
        chains = "all"
    write_pdb(
        gt_coords,
        atom_types,
        residue_types,
        residue_ids,
        f"{global_configs['infer']['results_dir']}/{global_configs['infer']['run_id']}/{ckpt_path_name}/{chains}/gt.pdb",
    )
    write_pdb(
        rec_coords,
        atom_types,
        residue_types,
        residue_ids,
        f"{global_configs['infer']['results_dir']}/{global_configs['infer']['run_id']}/{ckpt_path_name}/{chains}/rec.pdb",
    )
    # Save tokens
    test_outputs = {}
    for k in global_configs["infer"]["keys_to_summarize"] + global_configs["infer"]["keys_to_save"]:
        if k in batch:
            v = batch[k]
            if type(v) == torch.Tensor:
                if v.ndim > 1:
                    outputs_list = v[0][batch["eos_pad_mask"][0] == 0].detach().cpu().numpy().tolist()
                else:
                    outputs_list = v[0].detach().cpu().item()
            else:
                outputs_list = v
            test_outputs[k] = outputs_list
    with open(
        f"{global_configs['infer']['results_dir']}/{global_configs['infer']['run_id']}/{ckpt_path_name}/{chains}/outputs.json",
        "w",
    ) as f:
        json.dump(test_outputs, f, indent=4)


if __name__ == "__main__":
    main()
