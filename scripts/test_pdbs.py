#!/usr/bin/env python3
import os
import argparse
import tarfile
import tempfile
import zipfile
from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from bio2token.utils.lightning import find_lowest_val_loss_checkpoint
from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe, write_pdb
from bio2token.data.utils.molecule_conventions import ABBRS
import torch
import json
import glob
import pdb
DEBUG = False
if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def process_pdb(pdb_path, global_configs, model, device, ckpt_path_name, data_name):
    dict = pdb_2_dict(
        pdb_path,
        chains=global_configs["data"]["chains"] if "chains" in global_configs["data"] else None,
    )
    structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered, ca_indices = uniform_dataframe(
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
    # Inference
    batch = model(batch)
    for k, v in batch["losses"].items():
        batch[k] = v.item()
    # Save results
    gt_coords = batch["structure"][0].detach().cpu().numpy()
    rec_coords = batch["all_atom_coords"][0].detach().cpu().numpy()
    atom_types = atom_names_reordered
    residue_types = [ABBRS[res.split("_")[0]][res.split("_")[1]] for res in residue_name]
    if "chains" in global_configs["data"] and global_configs["data"]["chains"] is not None:
        chains = "_".join(global_configs["data"]["chains"])
    else:
        print("Using all chains")
        chains = "all"
    results_dir = f"{global_configs['infer']['results_dir']}/{global_configs['infer']['run_id']}/{ckpt_path_name}/{chains}/{data_name}/"
    os.makedirs(results_dir, exist_ok=True)
    pdb_basename = os.path.basename(pdb_path).replace(".pdb", "")
    # write_pdb(
    #     gt_coords,
    #     atom_types,
    #     residue_types,
    #     dict["res_ids"],
    #     dict["chains"],
    #     f"{results_dir}/gt_{pdb_basename}.pdb",
    # )
    # write_pdb(
    #     rec_coords,
    #     atom_types,
    #     residue_types,
    #     dict["res_ids"],
    #     dict["chains"],
    #     f"{results_dir}/rec_{pdb_basename}.pdb",
    # )
    # save only tokens, as .pt
    os.makedirs(f"{results_dir}/tokens", exist_ok=True)
    tokens = batch["indices"].detach().cpu()
    tokens = tokens.flatten()
    #torch.save(tokens, f"{results_dir}/tokens/{pdb_basename}.pt")
    ca_tokens = tokens[ca_indices]
    ca_gt_coords = gt_coords[ca_indices]
    
    tokens_str = ",".join(str(t.item()) for t in ca_tokens)
    coords_str = "#".join(";".join(f"{c:.3f}" for c in coord) for coord in ca_gt_coords)
    
    without_prefix = "_".join(pdb_basename.split("_")[1:])
    without_prefix = without_prefix.replace("-", "-TED")
    #torch.save(ca_tokens, f"{results_dir}/tokens/{without_prefix}_ca.pt")
    return dict["fasta_seqs"], tokens_str, coords_str, without_prefix
'''    # Save tokens
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
    with open(f"{results_dir}/outputs_{pdb_basename}.json", "w") as f:
        json.dump(test_outputs, f, indent=4)'''
def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_pdb.yaml")
    parser.add_argument("--pdb_dir", type=str, required=False, help="Directory containing .pdb files", default="/Users/lisaspindler/Downloads/casp15.targets.oligo.public_12.20.2022")
    args = parser.parse_args()
    # Load config yaml file
    global_configs = utilsyaml_to_dict(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)
    ckpt_path = f"{global_configs['infer']['checkpoint_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}/last.ckpt"
    if global_configs["infer"].get("checkpoint_type") == "best":
        ckpt_path = find_lowest_val_loss_checkpoint(
            checkpoint_dir=f"{global_configs['infer']['checkpoint_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}",
            checkpoint_monitor=global_configs["infer"]["checkpoint_monitor"],
            checkpoint_mode=global_configs["infer"]["checkpoint_mode"],
        )
    ckpt_path_name = ckpt_path.split("/")[-1].strip(".ckpt")
    state_dict = torch.load(ckpt_path)["state_dict"]
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    model.eval()
    model.to(device)
    all_fasta_seqs = {}
    label_data = {}
    data_name = os.path.basename(args.pdb_dir).replace(".zip", "")
    with tarfile.open(args.pdb_dir, "r:gz") as tarf:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_members = [m for m in tarf.getmembers() if m.name.endswith(".pdb")]
            print(f"Found {len(pdb_members)} .pdb files in tar archive: {args.pdb_dir}")
            if not pdb_members:
                print(f"No .pdb files found in tar archive: {args.pdb_dir}")
                return
            
            for member in pdb_members:
                extracted_path = os.path.join(tmpdir, os.path.basename(member.name))
                with tarf.extractfile(member) as pdb_file, open(extracted_path, "wb") as out_file:
                    out_file.write(pdb_file.read())

                fasta_dict, tokens_str, coords_str, header = process_pdb(
                    extracted_path, global_configs, model, device, ckpt_path_name, data_name
                )
                all_fasta_seqs.update(fasta_dict)
                label_data[header] = {
                    "tokens": tokens_str,
                    "coords": coords_str,
                }
    fasta_path = os.path.join("/dss/dsshome1/05/ge89maf2/results/bio2token/pdb_interactions_sample/sequences.fasta")
    with open(fasta_path, "w") as f:
        for header, seq in all_fasta_seqs.items():
            f.write(f">{header}\n{seq}\n")
    label_path = "/dss/dsshome1/05/ge89maf2/results/bio2token/pdb_interactions_sample/labels.fasta"
    with open(label_path, "w") as f:
        for header, data in label_data.items():
            f.write(f">{header}\n")
            f.write(f"{data['tokens']}\n")
            f.write(f"{data['coords']}\n")
    print(f"FASTA file written to: {fasta_path}")
if __name__ == "__main__":
    main()
    #uv run scripts/test_pdbs.py --config test_pdb.yaml --pdb_dir /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group2/sample_golden_dataset