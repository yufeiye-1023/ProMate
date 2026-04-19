#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data.py — ProMeta data utilities

Covers:
  - RDKit molecule → PyG graph conversion
  - SDF-based graph index construction
  - Protein sequence index construction
  - Label map construction
  - Episode sampling for meta-learning
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import rdchem

from torch_geometric.data import Data, Batch

from models import seq_to_tensor

# ── Atom feature set ──────────────────────────────────────────────────────────
# C  N  O  F  P  S  Cl  Br  I
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]

# E3 ligases treated as "rare" (held-out test targets)
RARE_E3_LIST = [
    "FEM1B", "IAP", "MDM2", "XIAP", "cIAP1",
    "DCAF1", "Keap1", "KLHL20", "DCAF16",
    "AhR", "BRD4", "RNF114", "KLHDC2",
    "DCAF11", "FBXO22", "UBR box", "RNF4",
]


# ── Molecule → graph ──────────────────────────────────────────────────────────

def _atom_features(atom: rdchem.Atom) -> List[float]:
    z = atom.GetAtomicNum()
    return [1.0 if z == x else 0.0 for x in ATOM_LIST]


def mol_to_graph(mol: rdchem.Mol) -> Optional[Data]:
    """Convert an RDKit Mol to a PyG Data object. Returns None on failure."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor(
        [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )
    edge_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list += [[i, j], [j, i]]
    if not edge_list:
        edge_list = [[0, 0]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def build_graph_index(sdf_path: str) -> Dict[str, Data]:
    """
    Read an SDF file and build a dict mapping Compound ID → PyG Data.

    The compound ID is read from the mol's '_Name' property, which
    PROTAC-DB SDF files store as the first line of each mol block.
    """
    print(f"📦 Loading SDF: {sdf_path}")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    index, ok, fail = {}, 0, 0
    for mol in suppl:
        if mol is None or not mol.HasProp("_Name"):
            fail += 1
            continue
        cid = str(mol.GetProp("_Name")).strip()
        g   = mol_to_graph(mol)
        if g is None:
            fail += 1
            continue
        index[cid] = g
        ok += 1
    print(f"✅ Graph index: {ok} ok | {fail} failed")
    return index


# ── Protein sequence index ────────────────────────────────────────────────────

def build_protein_index(
    seq_csv: str,
    seq_col: str = "POI_seq",
    max_len: int = 2000,
) -> Dict[str, torch.LongTensor]:
    """
    Read a CSV with columns ['Compound ID', seq_col] and return a dict:
        { compound_id_str : LongTensor(seq_indices) }

    Rows with missing / empty sequences are silently skipped.
    """
    print(f"📦 Building protein index from: {seq_csv}  (col='{seq_col}')")
    df = pd.read_csv(seq_csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)

    if seq_col not in df.columns:
        raise ValueError(
            f"Column '{seq_col}' not found in {seq_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    index, ok, fail = {}, 0, 0
    for _, row in df.iterrows():
        cid = str(row["Compound ID"])
        seq = str(row[seq_col]) if not pd.isna(row[seq_col]) else ""
        seq = seq.strip()
        if not seq or seq.lower() in ("nan", "none", ""):
            fail += 1
            continue
        index[cid] = seq_to_tensor(seq, max_len=max_len)
        ok += 1

    print(f"✅ Protein index: {ok} ok | {fail} skipped (missing seq)")
    return index


# ── Label map ─────────────────────────────────────────────────────────────────

def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """Return { compound_id_str : binary_label } from the dataframe."""
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df = df.copy()
        df["Label_bin"] = df["Label"].astype(int)
    return {
        str(row["Compound ID"]): int(row["Label_bin"])
        for _, row in df.iterrows()
        if not pd.isna(row.get("Label_bin"))
    }


# ── Episode helpers ───────────────────────────────────────────────────────────

def make_data_list(
    cids: List[str],
    graph_index: Dict[str, Data],
    label_map: Dict[str, int],
) -> List[Data]:
    """Retrieve graphs for a list of compound IDs, attaching labels and cids."""
    out = []
    for cid in [str(c) for c in cids]:
        if cid in graph_index and cid in label_map:
            g     = graph_index[cid].clone()
            g.y   = torch.tensor([label_map[cid]], dtype=torch.long)
            g.cid = cid
            out.append(g)
    return out


def build_meta_tasks(
    df: pd.DataFrame,
    graph_index: Dict[str, Data],
    K: int,
    Q: int,
    exclude_e3: Optional[Union[str, List[str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build meta-training tasks from the dataframe.

    Each task is a (E3 ligase, Target) group that has at least K+Q positives
    and K+Q negatives. Tasks belonging to excluded E3 ligases are omitted so
    that they can serve as held-out test targets.

    Args:
        exclude_e3: str, list of str, or None.
                    Pass "VHL" for CRBN→VHL experiments.
                    Pass RARE_E3_LIST (or use the helper below) for rare-E3.
    """
    df = df[df["Compound ID"].astype(str).isin(graph_index)].copy()

    if exclude_e3 is None:
        exclude_set = set()
    elif isinstance(exclude_e3, str):
        exclude_set = {exclude_e3}
    else:
        exclude_set = set(exclude_e3)

    tasks   = {}
    subset  = df[~df["E3 ligase"].isin(exclude_set)]
    need    = K + Q

    for (e3, tgt), sub in subset.groupby(["E3 ligase", "Target"]):
        sub  = sub.reset_index(drop=True)
        n_pos = int((sub["Label_bin"] == 1).sum())
        n_neg = int((sub["Label_bin"] == 0).sum())
        if n_pos >= need and n_neg >= need:
            tasks[f"{e3}__{tgt}"] = sub

    print(f"  E3s in meta-train: {sorted(subset['E3 ligase'].unique())}")
    return tasks


def sample_episode(
    df_task: pd.DataFrame,
    graph_index: Dict[str, Data],
    K: int,
    Q: int,
    device: torch.device,
    allow_replacement: bool = False,
) -> Optional[Tuple]:
    """
    Sample one episode from a task dataframe.

    Returns:
        (sup_data, sup_labels, qry_data, qry_labels) or None if sampling fails.
    """
    df_pos = df_task[df_task["Label_bin"] == 1]
    df_neg = df_task[df_task["Label_bin"] == 0]
    need   = K + Q

    if len(df_pos) == 0 or len(df_neg) == 0:
        return None
    rep_pos = allow_replacement and len(df_pos) < need
    rep_neg = allow_replacement and len(df_neg) < need
    if not allow_replacement and (len(df_pos) < need or len(df_neg) < need):
        return None

    pos = df_pos.sample(n=need, replace=rep_pos)
    neg = df_neg.sample(n=need, replace=rep_neg)

    rows_s = pd.concat([pos.iloc[:K], neg.iloc[:K]])
    rows_q = pd.concat([pos.iloc[K:], neg.iloc[K:]])

    def _to_tensors(rows):
        data_list, labels = [], []
        for _, r in rows.iterrows():
            cid = str(r["Compound ID"])
            if cid not in graph_index:
                return None, None
            g     = graph_index[cid].clone()
            g.cid = cid
            data_list.append(g)
            labels.append(int(r["Label_bin"]))
        return data_list, torch.tensor(labels, dtype=torch.float32, device=device)

    sup_data, sup_labels = _to_tensors(rows_s)
    qry_data, qry_labels = _to_tensors(rows_q)
    if sup_data is None or qry_data is None:
        return None

    return sup_data, sup_labels, qry_data, qry_labels
