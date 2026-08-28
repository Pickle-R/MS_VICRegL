#!/usr/bin/env python
"""Test STRICT de centre inédit : DRIAMS-D tenu à l'écart du pré-entraînement.

L'encodeur de runs/pretrain/ a été pré-entraîné en auto-supervision sur B∪C
*uniquement* (D n'existait pas encore dans data/). On NE ré-entraîne donc PAS :
on réutilise ce checkpoint et on évalue D comme un centre **jamais vu**, même pas
en non-labellisé — exactement la condition « hold-out DRIAMS-D » du design.

Protocole :
  - features = encodeur VICRegL GELÉ (runs/pretrain/ckpt.pt, vu seulement B∪C).
  - sonde linéaire (et baseline RF-binned) entraînées sur **B∪C poolés**,
    testées sur **D**  ->  condition phare  (B+C)->D.
  - conditions de contexte : B->D, C->D (centres séparés) et (B+C) in-domain 70/30
    (plafond). Espace de classes = top-N espèces communes à B, C ET D.

Sorties -> runs/holdout_D/ : comparison_holdout_D.json, RESULTS_holdout_D.md, figs/.

Usage :
    python scripts/08_holdout_D.py            # top_n=10
    python scripts/08_holdout_D.py --top_n 12 --run pretrain
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.analysis import (_bootstrap_ci, _mcnemar, _metrics,
                                  _rf_predict, _vicregl_predict)
from ms_vicregl.config import PROCESSED, RUNS, get_device
from ms_vicregl.dataset import load_center
from ms_vicregl.evaluate import common_topn_species, load_encoder
from ms_vicregl.pretrain import extract_features
from sklearn.metrics import balanced_accuracy_score, f1_score


def _eval(setting, kind, Ftr, ytr, Fte, yte, Xbtr, Xbte, n_classes,
          seed=0, n_boot=1000):
    pred_v = _vicregl_predict(Ftr, ytr, Fte)
    pred_r = _rf_predict(Xbtr, ytr, Xbte, seed=seed)
    rec = {"setting": setting, "kind": kind, "n_train": int(len(ytr)),
           "n_test": int(len(yte)), "n_classes": int(n_classes), "preds": {}}
    for name, pred in (("VICRegL", pred_v), ("RF-binned", pred_r)):
        m = _metrics(yte, pred)
        m["balanced_accuracy_CI95"] = _bootstrap_ci(
            yte, pred, balanced_accuracy_score, n=n_boot, seed=seed)
        m["f1_macro_CI95"] = _bootstrap_ci(
            yte, pred, lambda yt, yp: f1_score(yt, yp, average="macro",
                                               zero_division=0),
            n=n_boot, seed=seed)
        rec["preds"][name] = {"metrics": m}
    rec["mcnemar"] = _mcnemar(yte, pred_v, pred_r)
    return rec


def main(argv):
    ap = argparse.ArgumentParser(description="Hold-out DRIAMS-D (centre inédit)")
    ap.add_argument("--run", default="pretrain",
                    help="run du checkpoint encodeur (def: pretrain, vu B∪C)")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--variant", default="",
                    help='"" (défaut) ou "_snip" -- quelle entrée VICRegL charger '
                         '(cf. dataset.load_center); RF-binned inchangé')
    args = ap.parse_args(argv)

    if not (PROCESSED / "D_X.npy").exists():
        print("[hold-out D] data/processed/D_X.npy absent — lance d'abord "
              "`python scripts/01_ingest.py D`.")
        return

    device = get_device()
    enc = load_encoder(args.run)
    print(f"[hold-out D] encodeur={args.run} (pré-entraîné sur B∪C, D jamais vu) "
          f"| device={device}")

    data = {c: load_center(c, variant=args.variant) for c in ("B", "C", "D")}
    metas = [data[c][2] for c in ("B", "C", "D")]
    keep = common_topn_species(metas, args.top_n)
    le = LabelEncoder().fit(keep)
    print(f"[hold-out D] {len(keep)} espèces communes B∩C∩D : {keep}")

    feats, bins, ys = {}, {}, {}
    for c in ("B", "C", "D"):
        X, Xb, meta = data[c]
        mask = meta["species"].isin(keep).to_numpy()
        ys[c] = le.transform(meta.loc[mask, "species"].to_numpy())
        feats[c] = extract_features(enc, np.asarray(X)[mask], device=device)
        bins[c] = np.asarray(Xb)[mask]
        print(f"  {c}: {mask.sum()} spectres retenus")

    # B∪C poolés (train)
    F_bc = np.concatenate([feats["B"], feats["C"]])
    Xb_bc = np.concatenate([bins["B"], bins["C"]])
    y_bc = np.concatenate([ys["B"], ys["C"]])

    results = []
    # --- condition PHARE : (B+C) -> D (centre inédit) ---
    results.append(_eval("(B+C)->D", "hold-out", F_bc, y_bc, feats["D"], ys["D"],
                         Xb_bc, bins["D"], len(keep), args.seed, args.n_boot))
    # --- contexte : centres séparés ---
    results.append(_eval("B->D", "hold-out", feats["B"], ys["B"], feats["D"],
                         ys["D"], bins["B"], bins["D"], len(keep),
                         args.seed, args.n_boot))
    results.append(_eval("C->D", "hold-out", feats["C"], ys["C"], feats["D"],
                         ys["D"], bins["C"], bins["D"], len(keep),
                         args.seed, args.n_boot))
    # --- plafond : (B+C) in-domain 70/30 ---
    idx = np.arange(len(y_bc))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y_bc,
                              random_state=args.seed)
    results.append(_eval("(B+C) in-dom", "in-domain", F_bc[tr], y_bc[tr],
                         F_bc[te], y_bc[te], Xb_bc[tr], Xb_bc[te], len(keep),
                         args.seed, args.n_boot))

    out = {"run": args.run, "centers_train": ["B", "C"], "center_test": "D",
           "top_n": args.top_n, "species": keep, "seed": args.seed,
           "n_boot": args.n_boot, "device": str(device), "results": results}

    suffix = "" if args.run == "pretrain" else f"_{args.run}"
    run_dir = RUNS / f"holdout_D{suffix}"
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "comparison_holdout_D.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    _figure(out, run_dir / "figs" / "holdout_D_balacc.png")
    _report(out, run_dir / "RESULTS_holdout_D.md")

    print("\n=== Hold-out DRIAMS-D — balanced accuracy ===")
    for r in results:
        v = r["preds"]["VICRegL"]["metrics"]["balanced_accuracy"]
        rf = r["preds"]["RF-binned"]["metrics"]["balanced_accuracy"]
        flag = "  <-- CENTRE INÉDIT (phare)" if r["setting"] == "(B+C)->D" else ""
        print(f"  {r['setting']:13s} [{r['kind']:9s}] VICRegL {v:.3f} | "
              f"RF {rf:.3f}{flag}")
    head = results[0]
    print(f"\n>>> (B+C)->D : VICRegL bal-acc "
          f"{head['preds']['VICRegL']['metrics']['balanced_accuracy']:.3f} | "
          f"RF {head['preds']['RF-binned']['metrics']['balanced_accuracy']:.3f} "
          f"(n_test={head['n_test']}, {head['n_classes']} espèces)")
    print(f"Rapport -> {run_dir / 'RESULTS_holdout_D.md'}")


def _figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = out["results"]
    names = [r["setting"] for r in res]
    x = np.arange(len(names)); w = 0.38
    colors = {"VICRegL": "#2952cc", "RF-binned": "#d1603d"}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, pl in enumerate(("VICRegL", "RF-binned")):
        vals = [r["preds"][pl]["metrics"]["balanced_accuracy"] for r in res]
        cis = [r["preds"][pl]["metrics"]["balanced_accuracy_CI95"] for r in res]
        err = [[v - lo for v, (lo, hi) in zip(vals, cis)],
               [hi - v for v, (lo, hi) in zip(vals, cis)]]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=3, label=pl,
               color=colors[pl])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title("DRIAMS-D tenu à l'écart du pré-entraînement — VICRegL vs RF-binned")
    ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def _fmt_ci(m):
    lo, hi = m
    return f"[{lo:.3f}, {hi:.3f}]"


def _report(out, path):
    res = out["results"]
    L = ["# Test de généralisation à un centre inédit : DRIAMS-D tenu à l'écart "
         "du pré-entraînement\n",
         "## Résumé\n",
         "L'encodeur VICRegL a été pré-entraîné en auto-supervision sur **B∪C "
         "uniquement** ; **DRIAMS-D n'a jamais été vu**, même pas en non-labellisé. "
         "On réutilise ce checkpoint gelé et on entraîne la sonde linéaire (et la "
         "baseline RF-binned) sur **B∪C poolés**, puis on teste sur **D**. "
         "C'est le test de centre inédit le plus strict du protocole "
         "(contrairement à B↔C où les deux centres étaient vus en non-labellisé).\n",
         f"Espace de classes : {len(out['species'])} espèces communes à B, C et D — "
         "*" + "*, *".join(out["species"]) + "*.\n",
         "## Résultats — balanced accuracy (IC95 bootstrap)\n",
         "| Condition | Type | Pipeline | n_train | n_test | Bal-acc [IC95] | "
         "Accuracy | F1-macro | MCC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        for pl in ("VICRegL", "RF-binned"):
            m = r["preds"][pl]["metrics"]
            L.append(f"| {r['setting']} | {r['kind']} | {pl} | {r['n_train']} | "
                     f"{r['n_test']} | {m['balanced_accuracy']:.3f} "
                     f"{_fmt_ci(m['balanced_accuracy_CI95'])} | {m['accuracy']:.3f} | "
                     f"{m['f1_macro']:.3f} | {m['mcc']:.3f} |")
    L.append("")
    L.append("## Test apparié de McNemar (VICRegL vs RF-binned)\n")
    L.append("| Condition | VICRegL seul correct | RF seul correct | p-value | signif. |")
    L.append("|---|---|---|---|---|")
    for r in res:
        mc = r["mcnemar"]
        sig = ("***" if mc["pvalue"] < 1e-3 else "**" if mc["pvalue"] < 1e-2
               else "*" if mc["pvalue"] < 5e-2 else "ns")
        L.append(f"| {r['setting']} | {mc['only_A_correct']} | "
                 f"{mc['only_B_correct']} | {mc['pvalue']:.2e} | {sig} |")
    L.append("")
    head = next(r for r in res if r["setting"] == "(B+C)->D")
    vv = head["preds"]["VICRegL"]["metrics"]["balanced_accuracy"]
    rr = head["preds"]["RF-binned"]["metrics"]["balanced_accuracy"]
    L.append("## Lecture\n")
    L.append(f"Condition phare **(B+C)→D** (centre jamais vu) : VICRegL bal-acc "
             f"**{vv:.3f}** vs RF-binned **{rr:.3f}**. À comparer aux conditions "
             "B↔C de `runs/pretrain/RESULTS.md`, où l'encodeur avait été exposé "
             "(non labellisé) aux deux centres : l'écart mesure la perte de "
             "généralisation à un centre réellement inédit.\n")
    L.append("Figure : `figs/holdout_D_balacc.png`.\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
