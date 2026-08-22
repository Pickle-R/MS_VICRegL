"""Banc de prédiction de RÉSISTANCE (AMR) — comparable à Weis et al. 2022.

Pour chaque scénario (espèce, antibiotique), on prédit S vs R à partir du spectre :
  - features VICRegL gelées -> sonde logistique
  - binned_6000 -> Random Forest (baseline classique, cf. Weis)
Métrique = AUROC (celle de Weis). On mesure l'AUROC en INTRA-hôpital (CV 5-fold,
le meilleur cas de Weis) et en CROSS-hôpital (C<->B), puis la CHUTE cross-site —
le chiffre directement comparable à leur « drop 0.065–0.225 ».

Labels AMR lus dans les CSV id/ des tarballs (cachés en data/processed/{c}_id.parquet).
"""
from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA, PROCESSED, RUNS, get_device
from .dataset import load_center
from .evaluate import load_encoder
from .pretrain import extract_features

SCENARIOS = [
    ("Escherichia coli", "Ceftriaxone"),
    ("Escherichia coli", "Ciprofloxacin"),
    ("Staphylococcus aureus", "Oxacillin"),
    ("Klebsiella pneumoniae", "Ceftriaxone"),
]


def load_amr(center: str) -> pd.DataFrame:
    cache = PROCESSED / f"{center}_id.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    with tarfile.open(DATA / f"DRIAMS_{center}.tar.gz", "r:gz") as t:
        for m in t:
            if "/id/" in m.name and m.name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(t.extractfile(m).read()))
                df.to_parquet(cache)
                return df
    raise RuntimeError(f"id csv introuvable pour {center}")


def _scenario_data(center, species, drug, enc, device):
    """Retourne (features VICRegL, binned, y binaire R=1/S=0) pour un (espèce, drug)."""
    X, Xb, meta = load_center(center)
    idf = load_amr(center).drop_duplicates("code").set_index("code")
    if drug not in idf.columns:
        return None
    lab = idf[drug].reindex(meta["code"].values).astype(str).str.upper().values
    keep = (meta["species"] == species).values & np.isin(lab, ["S", "R"])
    if keep.sum() < 20:
        return None
    y = (lab[keep] == "R").astype(int)
    if y.sum() < 5 or (1 - y).sum() < 5:        # besoin des deux classes en nombre
        return None
    F = extract_features(enc, np.asarray(X)[keep], device=device)
    return {"F": F, "Xb": np.asarray(Xb)[keep], "y": y}


def _vic():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def _rf(seed=0):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                  class_weight="balanced", random_state=seed)


def _cv_auroc(model_fn, Xf, y, seed):
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 2:
        return None, None
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    proba = np.zeros(len(y))
    for tr, te in skf.split(Xf, y):
        m = model_fn(); m.fit(Xf[tr], y[tr]); proba[te] = m.predict_proba(Xf[te])[:, 1]
    return roc_auc_score(y, proba), average_precision_score(y, proba)


def _cross_auroc(model_fn, Xtr, ytr, Xte, yte):
    m = model_fn(); m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def run_amr(scenarios=SCENARIOS, run_name="pretrain", seed=0):
    device = get_device()
    enc = load_encoder(run_name)
    results = []
    for sp, drug in scenarios:
        d = {c: _scenario_data(c, sp, drug, enc, device) for c in ("C", "B")}
        if d["C"] is None or d["B"] is None:
            continue
        rec = {"species": sp, "drug": drug,
               "n_C": int(len(d["C"]["y"])), "nR_C": int(d["C"]["y"].sum()),
               "n_B": int(len(d["B"]["y"])), "nR_B": int(d["B"]["y"].sum())}
        for name, key, mfn in (("VICRegL", "F", _vic),
                               ("RF-binned", "Xb", lambda: _rf(seed))):
            inC = _cv_auroc(mfn, d["C"][key], d["C"]["y"], seed)[0]
            inB = _cv_auroc(mfn, d["B"][key], d["B"]["y"], seed)[0]
            cb = _cross_auroc(mfn, d["C"][key], d["C"]["y"], d["B"][key], d["B"]["y"])[0]
            bc = _cross_auroc(mfn, d["B"][key], d["B"]["y"], d["C"][key], d["C"]["y"])[0]
            indom = float(np.mean([inC, inB]))
            cross = float(np.mean([cb, bc]))
            rec[name] = {"in_C": inC, "in_B": inB, "in_domain": indom,
                         "cross_CtoB": cb, "cross_BtoC": bc, "cross": cross,
                         "drop": indom - cross}
        results.append(rec)

    out = {"run": run_name, "seed": seed, "device": str(device),
           "weis_ref": {"E.coli/Ceftriaxone": 0.75, "K.pneumoniae/Ceftriaxone": 0.75,
                        "S.aureus/Oxacillin": 0.82, "cross_site_drop": [0.065, 0.225]},
           "results": results}
    run_dir = RUNS / run_name
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "amr.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    _fig(out, run_dir / "figs")
    _report(out, run_dir / "RESULTS_amr.md")
    return out


def _fig(out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = out["results"]
    names = [f"{r['species'].split()[0][0]}.{r['species'].split()[1][:4]}\n{r['drug'][:6]}"
             for r in res]
    x = np.arange(len(names)); w = 0.2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    # AUROC cross-hôpital
    for i, (pl, col) in enumerate((("VICRegL", "#2952cc"), ("RF-binned", "#d1603d"))):
        a1.bar(x + (i - 0.5) * w * 2, [r[pl]["cross"] for r in res], w * 2,
               label=pl, color=col)
    a1.axhline(0.5, ls=":", color="grey"); a1.set_ylim(0.4, 1.0)
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=8)
    a1.set_ylabel("AUROC cross-hôpital"); a1.set_title("AUROC en transfert inter-hôpitaux")
    a1.legend()
    # chute cross-site vs bande Weis
    a2.axhspan(0.065, 0.225, color="grey", alpha=0.2, label="drop Weis (0.065–0.225)")
    for i, (pl, col) in enumerate((("VICRegL", "#2952cc"), ("RF-binned", "#d1603d"))):
        a2.bar(x + (i - 0.5) * w * 2, [r[pl]["drop"] for r in res], w * 2,
               label=pl, color=col)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=8)
    a2.set_ylabel("Chute AUROC (intra − cross)")
    a2.set_title("Instabilité inter-hôpitaux (plus bas = mieux)"); a2.legend(fontsize=8)
    fig.suptitle("Banc AMR — VICRegL vs RF-binned (réf. Weis et al. 2022)")
    fig.tight_layout(); fig.savefig(figdir / "amr_auroc.png", dpi=150); plt.close(fig)


def _report(out, path):
    res = out["results"]
    L = ["# Banc de prédiction de résistance (AMR) — VICRegL vs RF-binned, "
         "référence Weis et al. 2022\n",
         "Sonde sur features VICRegL gelées vs Random Forest sur binned_6000. "
         "Métrique AUROC (celle de Weis). Intra-hôpital = CV 5-fold (meilleur cas) ; "
         "cross-hôpital = train un centre / test l'autre (C↔B). La **chute** "
         "intra→cross est le chiffre directement comparable à Weis "
         "(**drop rapporté : 0.065–0.225**).\n",
         "> Réf. Weis et al. (DRIAMS, *Nat Med* 2022) : AUROC ≈ 0.75 (ceftriaxone/*E. coli* "
         "et /*K. pneumoniae*), 0.82 (oxacilline/*S. aureus*).\n"]
    L.append("## AUROC par scénario\n")
    L.append("| Scénario | n (C/B) | R (C/B) | Pipeline | AUROC intra | AUROC cross | chute |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res:
        for pl in ("VICRegL", "RF-binned"):
            m = r[pl]
            L.append(f"| *{r['species']}* / {r['drug']} | {r['n_C']}/{r['n_B']} | "
                     f"{r['nR_C']}/{r['nR_B']} | {pl} | {m['in_domain']:.3f} | "
                     f"{m['cross']:.3f} | {m['drop']:+.3f} |")
    L.append("")
    L.append("## Synthèse stabilité (chute AUROC intra→cross, moyenne)\n")
    L.append("| Pipeline | chute moyenne | vs bande Weis (0.065–0.225) |")
    L.append("|---|---|---|")
    for pl in ("VICRegL", "RF-binned"):
        drops = [r[pl]["drop"] for r in res]
        md = float(np.mean(drops))
        comp = "dans/sous la bande" if md <= 0.225 else "au-dessus"
        L.append(f"| {pl} | {md:+.3f} | {comp} |")
    L.append("\n## Lecture\n")
    L.append("La prédiction AMR est intrinsèquement plus difficile que l'ID d'espèce "
             "(la résistance tient souvent à un gène sans signature spectrale nette) : les "
             "AUROC absolus sont modestes, conformes à Weis. Le point d'intérêt est la "
             "**stabilité inter-hôpitaux** : si VICRegL présente une chute cross-site plus "
             "faible que le RF-binned, l'invariance apprise bénéficie aussi à l'AMR — "
             "prolongeant la solution démontrée sur l'espèce au problème que Weis et al. "
             "avaient identifié comme limite principale.\n")
    path.write_text("\n".join(L))
