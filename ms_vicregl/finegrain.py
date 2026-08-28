"""Discrimination FINE d'espèces proches (cas difficiles du MALDI-TOF).

Pour chaque groupe d'espèces phylogénétiquement proches, on mesure la séparabilité
intrinsèque (et non plus la robustesse de centre) : encodeur VICRegL **gelé** ->
validation croisée stratifiée 5-fold sur B∪C (effectifs petits : la CV utilise toutes
les données), sonde linéaire VICRegL vs Random Forest sur binned_6000.

⚠️ Mise en garde : les labels DRIAMS sont produits par MALDI (Bruker Biotyper), pas par
NGS. Sur ces espèces proches, ils sont donc partiellement faillibles : on mesure la
séparabilité « telle qu'étiquetée par Biotyper », pas une vérité confirmée par séquençage.

Sortie : runs/pretrain/finegrain.json, RESULTS_finegrain.md, figs/finegrain_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

from .config import RUNS, get_device
from .dataset import load_centers
from .evaluate import load_encoder
from .pretrain import extract_features

GROUPS = {
    "Enterobacter cloacae complex": [
        "Enterobacter cloacae", "Enterobacter aerogenes", "Enterobacter asburiae",
        "Enterobacter kobei", "Enterobacter ludwigii"],
    "Streptococcus viridans (pneumoniae/oralis/mitis)": [
        "Streptococcus pneumoniae", "Streptococcus oralis", "Streptococcus mitis"],
    "Klebsiella (pneumoniae/oxytoca/variicola)": [
        "Klebsiella pneumoniae", "Klebsiella oxytoca", "Klebsiella variicola"],
}


def _abbr(s):
    p = s.split()
    return f"{p[0][0]}. {p[1]}" if len(p) >= 2 else s


def _metrics(y, p):
    return {"accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y, p)}


def _oof(F, Xb, y, n_splits, seed):
    """Prédictions out-of-fold (VICRegL & RF) par CV stratifiée."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ov, orf = np.empty_like(y), np.empty_like(y)
    for tr, te in skf.split(F, y):
        probe = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        probe.fit(F[tr], y[tr]); ov[te] = probe.predict(F[te])
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                    class_weight="balanced", random_state=seed)
        rf.fit(Xb[tr], y[tr]); orf[te] = rf.predict(Xb[te])
    return ov, orf


def _mcnemar(y, pa, pb):
    ca, cb = (pa == y), (pb == y)
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    table = [[int(np.sum(ca & cb)), b], [c, int(np.sum(~ca & ~cb))]]
    res = mcnemar(table, exact=(b + c) < 25, correction=True)
    return {"only_VICRegL": b, "only_RF": c,
            "pvalue": float(res.pvalue), "statistic": float(res.statistic)}


def run_finegrain(run_name="pretrain", seed=0, n_splits=5, n_segments=1):
    """n_segments=1 (comportement historique, RESULT 2) : sur ces petits groupes
    (quelques centaines d'échantillons), la résolution spatiale max ("max") a été
    testée et fait légèrement RÉGRESSER 2/3 groupes (sur-apprentissage de la sonde
    en haute dimension) -- contrairement à l'ID cross-centre (analysis.py) où elle
    aide nettement. Passer n_segments="max" explicitement pour comparer."""
    device = get_device()
    enc = load_encoder(run_name)
    X, Xb, meta = load_centers(["C", "B"])
    sp_all = meta["species"].to_numpy()

    groups_out = []
    for title, species in GROUPS.items():
        mask = np.isin(sp_all, species)            # exclut d'office les labels 'MIX!...'
        if mask.sum() < 30:
            continue
        le = LabelEncoder().fit([s for s in species if s in set(sp_all[mask])])
        y = le.transform(sp_all[mask])
        F = extract_features(enc, X[mask], device=device, n_segments=n_segments)
        Xbg = Xb[mask]
        counts = {le.inverse_transform([k])[0]: int((y == k).sum())
                  for k in range(len(le.classes_))}
        k = min(n_splits, min(counts.values()))
        ov, orf = _oof(F, Xbg, y, k, seed)
        rec = {"group": title, "classes": list(le.classes_), "counts": counts,
               "n": int(mask.sum()), "n_splits": k,
               "VICRegL": _metrics(y, ov), "RF-binned": _metrics(y, orf),
               "mcnemar": _mcnemar(y, ov, orf),
               "cm_VICRegL": confusion_matrix(y, ov, normalize="true").tolist(),
               "cm_RF": confusion_matrix(y, orf, normalize="true").tolist(),
               "labels": [_abbr(s) for s in le.classes_]}
        groups_out.append(rec)

    out = {"run": run_name, "seed": seed, "device": str(device),
           "centers": ["C", "B"], "groups": groups_out}
    run_dir = RUNS / run_name
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "finegrain.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    _figs(out, run_dir / "figs")
    _report(out, run_dir / "RESULTS_finegrain.md")
    return out


def _figs(out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # matrices de confusion par groupe
    for gi, g in enumerate(out["groups"]):
        labs = g["labels"]
        fig, axes = plt.subplots(1, 2, figsize=(2.2 + 1.1 * len(labs), 4.6))
        for ax, pl, key in zip(axes, ("VICRegL", "RF-binned"), ("cm_VICRegL", "cm_RF")):
            cm = np.array(g[key])
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{pl}\nbal-acc={g[pl]['balanced_accuracy']:.3f}", fontsize=10)
            ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
            ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(labs, fontsize=8)
            ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
            for i in range(len(labs)):
                for j in range(len(labs)):
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(g["group"], fontsize=11)
        fig.tight_layout()
        fig.savefig(figdir / f"finegrain_cm_{gi}.png", dpi=150); plt.close(fig)

    # barres récapitulatives balanced accuracy
    names = [g["group"].split(" (")[0] for g in out["groups"]]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, pl in enumerate(("VICRegL", "RF-binned")):
        vals = [g[pl]["balanced_accuracy"] for g in out["groups"]]
        ax.bar(x + (i - 0.5) * w, vals, w, label=pl,
               color="#2952cc" if pl == "VICRegL" else "#d1603d")
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Balanced accuracy (CV 5-fold)"); ax.set_ylim(0, 1.08)
    ax.set_title("Discrimination fine d'espèces proches — VICRegL vs RF-binned")
    ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "finegrain_summary.png", dpi=150); plt.close(fig)


def _report(out, path):
    L = ["# Discrimination fine d'espèces phylogénétiquement proches "
         "(cas difficiles du MALDI-TOF)\n",
         "Encodeur VICRegL **gelé** (pré-entraîné sur B∪C non labellisés), évaluation par "
         "**validation croisée stratifiée 5-fold** sur B∪C. Sonde linéaire VICRegL vs Random "
         "Forest sur binned_6000.\n",
         "> ⚠️ **Mise en garde gold standard.** Les labels DRIAMS proviennent d'une "
         "identification **MALDI (Bruker Biotyper)**, non du **NGS**. Sur ces espèces proches, "
         "ils sont partiellement faillibles : on mesure la séparabilité *telle qu'étiquetée par "
         "Biotyper*, pas une vérité confirmée par séquençage. C'est précisément la limite que "
         "MSclassifR levait en utilisant le NGS comme étalon-or.\n"]
    L.append("## Synthèse (balanced accuracy, CV 5-fold)\n")
    L.append("| Groupe | n | classes | VICRegL | RF-binned | McNemar p | meilleur |")
    L.append("|---|---|---|---|---|---|---|")
    for g in out["groups"]:
        v = g["VICRegL"]["balanced_accuracy"]; r = g["RF-binned"]["balanced_accuracy"]
        best = "VICRegL" if v > r else "RF-binned" if r > v else "—"
        L.append(f"| {g['group']} | {g['n']} | {len(g['classes'])} | **{v:.3f}** | "
                 f"{r:.3f} | {g['mcnemar']['pvalue']:.1e} | {best} |")
    L.append("")
    for g in out["groups"]:
        L.append(f"## {g['group']}\n")
        L.append("Effectifs : " + ", ".join(f"*{k}* {n}" for k, n in g["counts"].items()) + ".\n")
        L.append("| Pipeline | balanced-acc | accuracy | F1-macro | MCC |")
        L.append("|---|---|---|---|---|")
        for pl in ("VICRegL", "RF-binned"):
            m = g[pl]
            L.append(f"| {pl} | {m['balanced_accuracy']:.3f} | {m['accuracy']:.3f} | "
                     f"{m['f1_macro']:.3f} | {m['mcc']:.3f} |")
        mc = g["mcnemar"]
        L.append(f"\nMcNemar : VICRegL seul correct = {mc['only_VICRegL']}, RF seul correct = "
                 f"{mc['only_RF']}, p = {mc['pvalue']:.2e}.\n")
        L.append(f"Matrice de confusion : `figs/finegrain_cm_{out['groups'].index(g)}.png`.\n")
    L.append("## Lecture\n")
    L.append("Ces tests mesurent la **séparabilité intrinsèque** d'espèces aux empreintes "
             "protéiques quasi identiques — un problème distinct de la robustesse inter-centres. "
             "Une performance imparfaite peut refléter soit une limite physique du MALDI, soit "
             "des labels Biotyper erronés (non corrigés faute de NGS). Pour une conclusion "
             "définitive au rang d'espèce, un sous-ensemble **labellisé par séquençage** serait "
             "nécessaire.\n")
    path.write_text("\n".join(L))
