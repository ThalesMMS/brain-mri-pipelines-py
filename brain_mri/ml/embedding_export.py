import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None


def _export_embeddings(*, model, dataset_obj, device, batch_size: int, mode: str, backbone: str, output_dir: Path):
    """
    Create and save CSVs of model embeddings and targets for a dataset split.
    
    Parameters:
        model: A trained model that provides either an `extract_features(axl, cor, sag, clin)` method or encoder/projection attributes (`enc_axl`, `enc_cor`, `enc_sag`, `proj_axl`, `proj_cor`, `proj_sag`) and optional `tab_mlp`.
        dataset_obj: Dataset-like object supporting len(), iteration returning (batch_x, labels), a pandas `df` with per-row metadata (used to obtain `MRI_ID` and, for regression, `age`), and an optional `_split_name` attribute.
        device: Torch device to move tensors to for embedding extraction.
        batch_size (int): Batch size for the DataLoader used to extract embeddings.
        mode (str): Either `"classification"` or `"regression"`, controls which target is written to the CSV.
        backbone (str): Backbone identifier used to name the output CSV file.
        output_dir (Path): Directory where the embeddings CSV will be written.
    
    Returns:
        Path or None: Path to the written CSV file, or `None` if the dataset is empty.
    """
    if len(dataset_obj) == 0:
        return None

    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)
    emb_list, target_list, ids = [], [], []
    model.eval()
    idx_offset = 0
    with torch.no_grad():
        for batch_x, lbls in loader:
            axl = batch_x["axl"].to(device)
            cor = batch_x["cor"].to(device)
            sag = batch_x["sag"].to(device)
            clin = batch_x.get("clin")
            if clin is not None:
                clin = clin.to(device)

            if hasattr(model, "extract_features"):
                feats = model.extract_features(axl, cor, sag, clin)
            else:
                f_axl = model._encode_one(model.enc_axl, model.proj_axl, axl)
                f_cor = model._encode_one(model.enc_cor, model.proj_cor, cor)
                f_sag = model._encode_one(model.enc_sag, model.proj_sag, sag)
                feats_list = [f_axl, f_cor, f_sag]
                if clin is not None and model.tab_mlp:
                    feats_list.append(model.tab_mlp(clin))
                feats = torch.cat(feats_list, dim=1)

            emb_list.append(feats.cpu().numpy())
            target_list.append(lbls.cpu().numpy())
            rows = dataset_obj.df.iloc[idx_offset : idx_offset + len(lbls)]
            ids.extend(rows.get("MRI_ID", rows.index).tolist())
            idx_offset += len(lbls)

    emb_arr = np.concatenate(emb_list)
    tgt_arr = np.concatenate(target_list)
    split_name = getattr(dataset_obj, "_split_name", "split")
    df_emb = pd.DataFrame(emb_arr)
    df_emb.insert(0, "MRI_ID", ids)
    if mode == "regression" and "age" in dataset_obj.df.columns:
        df_emb["target"] = dataset_obj.df["age"].iloc[: len(df_emb)].values
    else:
        df_emb["target"] = tgt_arr
    out_path = output_dir / f"{backbone}_embeddings_{mode}_{split_name}.csv"
    df_emb.to_csv(out_path, index=False)
    return out_path
