#!/usr/bin/env python3
from datetime import datetime
import os
import argparse
import zipfile
import io
from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from bio2token.utils.lightning import find_lowest_val_loss_checkpoint
from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe, write_pdb
import torch

def process_pdb_batch(pdb_file_objs, pdb_names, global_configs, device):
    # Prepare lists for batch processing
    batch_structures = []
    batch_unknown_structures = []
    batch_residue_ids = []
    batch_token_class = []
    batch_ca_indices = []
    batch_atom_names_reordered = []
    batch_residue_names = []
    batch_dicts = []
    batch_fasta_dicts = []
    batch_pdb_basenames = []

    for pdb_file_obj, pdb_name in zip(pdb_file_objs, pdb_names):
        dict = pdb_2_dict(
            pdb_file_obj,
            pdb_name,
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
        batch_structures.append(torch.tensor(structure).float())
        batch_unknown_structures.append(torch.tensor(unknown_structure).bool())
        batch_residue_ids.append(torch.tensor(residue_ids).long())
        batch_token_class.append(torch.tensor(token_class).long())
        batch_ca_indices.append(ca_indices)
        batch_atom_names_reordered.append(atom_names_reordered)
        batch_residue_names.append(residue_name)
        batch_dicts.append(dict)
        batch_fasta_dicts.append(dict["fasta_seqs"])
        batch_pdb_basenames.append(os.path.basename(pdb_name).replace(".pdb", ""))

    # Padding
    from torch.nn.utils.rnn import pad_sequence

    # Find max length for padding
    max_len = max([s.shape[0] for s in batch_structures])

    def pad_tensor_list(tensor_list, pad_value=0):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.shape[0],) + t.shape[1:], pad_value, dtype=t.dtype)]) if t.shape[0] < max_len else t
            for t in tensor_list
        ])

    structure = pad_tensor_list(batch_structures, pad_value=0)
    unknown_structure = pad_tensor_list(batch_unknown_structures, pad_value=True)
    residue_ids = pad_tensor_list(batch_residue_ids, pad_value=0)
    token_class = pad_tensor_list(batch_token_class, pad_value=0)

    batch = {
        "structure": structure,
        "unknown_structure": unknown_structure,
        "residue_ids": residue_ids,
        "token_class": token_class,
    }
    # NICHT flatten!
    # batch = {k: v[~batch["unknown_structure"]] if k != "unknown_structure" else v for k, v in batch.items()}
    batch = compute_masks(batch, structure_track=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    return batch, batch_ca_indices, batch_atom_names_reordered, batch_residue_names, batch_dicts, batch_fasta_dicts, batch_pdb_basenames

def process_zip(zip_path, args, global_configs, model, device):
    input_base = os.path.splitext(os.path.basename(zip_path))[0]
    fasta_path = os.path.join(args.data_dir, f"{input_base}_predictions.fasta")
    label_path = os.path.join(args.data_dir, f"{input_base}_predictions.labels")

    if os.path.exists(fasta_path) and os.path.exists(label_path):
        print(f"✅ Skipping {zip_path} (results already exist)")
        return

    print(f"\n🔄 Processing {zip_path} ...")
    start_zip = datetime.now()
    all_fasta_seqs = {}
    label_data = {}
    data_name = input_base

    with zipfile.ZipFile(zip_path, "r") as zipf:
        print(f"  ➡️ Processing zip archive: {zip_path}")
        pdb_names_in_zip = [name for name in zipf.namelist() if name.endswith(".pdb")]
        print(f"  ➡️ Found {len(pdb_names_in_zip)} .pdb files in zip archive: {zip_path}")
        if not pdb_names_in_zip:
            print("  ❌ No .pdb files found.")
            return

        for i in range(0, len(pdb_names_in_zip), args.batch_size):
            batch_names = pdb_names_in_zip[i:i+args.batch_size]
            print(f"    ...Batch {i//args.batch_size+1}/{(len(pdb_names_in_zip)-1)//args.batch_size+1} ({len(batch_names)} files)")
            batch_start = datetime.now()
            pdb_file_objs, pdb_file_names = [], []
            for name in batch_names:
                with zipf.open(name) as f:
                    pdb_str = f.read().decode("utf-8")
                    pdb_file_objs.append(io.StringIO(pdb_str))
                    pdb_file_names.append(name)

            batch, batch_ca_indices, batch_atom_names_reordered, batch_residue_names, batch_dicts, batch_fasta_dicts, batch_pdb_basenames = process_pdb_batch(
                pdb_file_objs, pdb_file_names, global_configs, device
            )
            with torch.no_grad():
                batch_out = model(batch)
            for j, pdb_basename in enumerate(batch_pdb_basenames):
                dict = batch_dicts[j]
                ca_indices = batch_ca_indices[j]
                atom_names_reordered = batch_atom_names_reordered[j]
                residue_name = batch_residue_names[j]
                fasta_dict = batch_fasta_dicts[j]
                gt_coords = batch_out["structure"][j].detach().cpu().numpy()
                tokens = batch_out["indices"][j].detach().cpu().flatten()
                ca_tokens = tokens[ca_indices]
                ca_gt_coords = gt_coords[ca_indices]
                tokens_str = ",".join(str(t.item()) for t in ca_tokens)
                coords_str = "#".join(";".join(f"{c:.3f}" for c in coord) for coord in ca_gt_coords)
                header = pdb_basename.replace("-", "-TED")
                all_fasta_seqs.update(fasta_dict)
                label_data[header] = {
                    "tokens": tokens_str,
                    "coords": coords_str,
                }
            batch_end = datetime.now()
            print(f"    ...Batch done in {batch_end-batch_start}")

    os.makedirs(args.data_dir, exist_ok=True)
    with open(fasta_path, "w") as f:
        for header, seq in all_fasta_seqs.items():
            f.write(f">{header}\n{seq}\n")
    with open(label_path, "w") as f:
        for header, data in label_data.items():
            f.write(f">{header}\n{data['tokens']}\n{data['coords']}\n")

    end_zip = datetime.now()
    print(f"✅ Done {zip_path} in {end_zip-start_zip}")
    print(f"   FASTA:  {fasta_path}")
    print(f"   LABELS: {label_path}")

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_pdb.yaml")
    parser.add_argument("--zip_dir", type=str, required=True, help="Directory containing .zip files")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to save output files")
    args = parser.parse_args()
    print("Load everything!")
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

    # Alle ZIPs im Verzeichnis finden
    zip_files = [os.path.join(args.zip_dir, f) for f in os.listdir(args.zip_dir) if f.endswith(".zip")]
    print(f"\nGefundene ZIP-Dateien: {len(zip_files)}")
    print("\n".join([f"  {os.path.basename(z)}" for z in zip_files]))
    print()

    total_start = datetime.now()
    for idx, zip_path in enumerate(zip_files):
        print(f"\n{'='*40}\n[{idx+1}/{len(zip_files)}] {os.path.basename(zip_path)}")
        process_zip(zip_path, args, global_configs, model, device)
    total_end = datetime.now()
    print(f"\n✅ Alle ZIPs fertig in {total_end-total_start}")

if __name__ == "__main__":
    start = datetime.now()
    print("Start!")
    main()
    end = datetime.now()
    print(f"⏱ Gesamtlaufzeit: {end - start}")
