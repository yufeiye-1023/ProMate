#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py — ProMeta evaluation entry point

Loads trained encoder checkpoints and evaluates ProMeta on a held-out
episodic split via prototype-based classification (no parameter updates).

Example:
    python predict.py \\
        --csv      protac_filtered_balanced.csv \\
        --sdf      protac.sdf \\
        --seq-csv  protac_with_seq.csv \\
        --split    splits/crbn_vhl_K2Q3_seed42.json \\
        --encoder  results/vhl_K2Q3_seed42/encoder_seed42.pt \\
        --poi-encoder results/vhl_K2Q3_seed42/poi_encoder_seed42.pt \\
        --e3-encoder  results/vhl_K2Q3_seed42/e3_encoder_seed42.pt \\
        --seed     42 \\
        --outdir   results/vhl_K2Q3_seed42
"""

import json
import os
import argparse

import pandas as pd
import torch

from torch_geometric.data import Batch

from models import GCNEncoder, ProteinEncoder
from data import (
    build_graph_index,
    build_protein_index,
    build_label_map,
    make_data_list,
)
from utils import set_seed, metrics_from_logits, summarize_records, mean_ignore_nan


# ── Encoding helpers (mirrors train.py) ───────────────────────────────────────

def _get_cids(data_list):
    return [getattr(g, "cid", "") for g in data_list]


def encode_batch(encoder, data_list, device):
    b = Batch.from_data_list(data_list).to(device)
    return encoder(b.x, b.edge_index, b.batch)


def encode_batch_fused(encoder, poi_enc, e3_enc, data_list,
                       poi_index, e3_index, device, out_dim):
    b     = Batch.from_data_list(data_list).to(device)
    z_mol = encoder(b.x, b.edge_index, b.batch)
    cids  = _get_cids(data_list)
    z_poi = poi_enc.encode_batch_ids(cids, poi_index, device, out_dim)
    z_e3  = e3_enc.encode_batch_ids( cids, e3_index,  device, out_dim)
    return z_mol + poi_enc.alpha * z_poi + e3_enc.alpha * z_e3


def make_encode_fn(encoder, poi_enc, e3_enc, poi_index, e3_index, device, out_dim):
    if poi_index is not None:
        return lambda dl: encode_batch_fused(
            encoder, poi_enc, e3_enc, dl, poi_index, e3_index, device, out_dim
        )
    return lambda dl: encode_batch(encoder, dl, device)


# ── Prototypical inference ────────────────────────────────────────────────────

@torch.no_grad()
def prototypical_logits(encode_fn, sup_data, sup_labels, qry_data):
    z_s = encode_fn(sup_data)
    z_q = encode_fn(qry_data)
    pp  = z_s[sup_labels == 1].mean(0, keepdim=True)
    pn  = z_s[sup_labels == 0].mean(0, keepdim=True)
    return -(torch.norm(z_q - pp, dim=1) - torch.norm(z_q - pn, dim=1))


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    encoder, poi_enc, e3_enc,
    split_path, graph_index, label_map,
    poi_index, e3_index, device, out_dim,
    phase="meta_test",
):
    print(f"🧪 Evaluating ProMeta | phase={phase}")
    encoder.eval()
    if poi_enc is not None: poi_enc.eval()
    if e3_enc  is not None: e3_enc.eval()

    encode_fn = make_encode_fn(
        encoder, poi_enc, e3_enc, poi_index, e3_index, device, out_dim
    )
    phase_obj = json.load(open(split_path)).get(phase, {})
    records, skipped, total = [], 0, 0

    for task, episodes in phase_obj.items():
        for ep_idx, ep in enumerate(episodes):
            total += 1
            sup_data = make_data_list(ep.get("support", []), graph_index, label_map)
            qry_data = make_data_list(ep.get("query",   []), graph_index, label_map)

            if not sup_data or not qry_data:
                skipped += 1
                continue

            sup_labels = torch.tensor(
                [g.y.item() for g in sup_data], dtype=torch.float32, device=device
            )
            qry_labels = torch.tensor(
                [g.y.item() for g in qry_data], dtype=torch.float32, device=device
            )

            if not ((sup_labels == 1).any() and (sup_labels == 0).any()):
                skipped += 1
                continue

            logits = prototypical_logits(encode_fn, sup_data, sup_labels, qry_data)
            rec    = {"Task": task, "Episode": ep_idx}
            rec.update(metrics_from_logits(logits, qry_labels))
            records.append(rec)

    import pandas as pd
    print(f"✅ valid={len(records)} | skipped={skipped} | total={total}")
    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="ProMeta — evaluation")

    # Data
    ap.add_argument("--csv",     required=True)
    ap.add_argument("--sdf",     required=True)
    ap.add_argument("--split",   required=True)
    ap.add_argument("--phase",   default="meta_test",
                    choices=["meta_test", "meta_valid", "meta_train"])
    ap.add_argument("--seq-csv",    default="")
    ap.add_argument("--seq-col",    default="POI_seq")
    ap.add_argument("--e3-seq-col", default="E3_seq")

    # Checkpoints
    ap.add_argument("--encoder",     required=True,
                    help="Path to encoder_seed{N}.pt")
    ap.add_argument("--poi-encoder", default="",
                    help="Path to poi_encoder_seed{N}.pt (optional)")
    ap.add_argument("--e3-encoder",  default="",
                    help="Path to e3_encoder_seed{N}.pt (optional)")

    # Output
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", default="cuda")

    # Encoder architecture (must match training config)
    ap.add_argument("--encoder-hidden",  type=int,   default=128)
    ap.add_argument("--encoder-out",     type=int,   default=128)
    ap.add_argument("--encoder-layers",  type=int,   default=3)
    ap.add_argument("--encoder-dropout", type=float, default=0.1)
    ap.add_argument("--prot-embed-dim",  type=int,   default=64)
    ap.add_argument("--prot-max-len",    type=int,   default=2000)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device  = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    out_dim = args.encoder_out

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df["Label_bin"] = df["Label"].astype(int)

    graph_index = build_graph_index(args.sdf)
    label_map   = build_label_map(df)

    # ── Protein index ──────────────────────────────────────────────────────────
    poi_index, e3_index, poi_enc, e3_enc = None, None, None, None
    if args.seq_csv:
        poi_index = build_protein_index(args.seq_csv, args.seq_col,    args.prot_max_len)
        e3_index  = build_protein_index(args.seq_csv, args.e3_seq_col, args.prot_max_len)
        poi_enc   = ProteinEncoder(args.prot_embed_dim, out_dim).to(device)
        e3_enc    = ProteinEncoder(args.prot_embed_dim, out_dim).to(device)

    # ── Load encoder checkpoints ───────────────────────────────────────────────
    encoder = GCNEncoder(
        in_dim=9,
        hidden_dim=args.encoder_hidden,
        out_dim=out_dim,
        num_layers=args.encoder_layers,
        dropout=args.encoder_dropout,
    ).to(device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    print(f"📥 Encoder loaded ← {args.encoder}")

    if poi_enc is not None and args.poi_encoder:
        poi_enc.load_state_dict(torch.load(args.poi_encoder, map_location=device))
        print(f"📥 POI encoder loaded ← {args.poi_encoder}")
    if e3_enc is not None and args.e3_encoder:
        e3_enc.load_state_dict(torch.load(args.e3_encoder, map_location=device))
        print(f"📥 E3  encoder loaded ← {args.e3_encoder}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    records = evaluate(
        encoder, poi_enc, e3_enc,
        args.split, graph_index, label_map,
        poi_index, e3_index, device, out_dim,
        phase=args.phase,
    )

    if records.empty:
        raise RuntimeError("No valid evaluation records produced.")

    ep_csv  = os.path.join(args.outdir, f"episodes_seed{args.seed}.csv")
    sum_csv = os.path.join(args.outdir, f"summary_seed{args.seed}.csv")
    records.to_csv(ep_csv, index=False)
    summary = summarize_records(records, "ProMeta")
    summary.to_csv(sum_csv, index=False)

    print(f"\n📝 Episodes → {ep_csv}")
    print(f"📝 Summary  → {sum_csv}")
    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
