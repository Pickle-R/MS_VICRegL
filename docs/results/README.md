# Résultats bruts (checkpoints exclus)

Copie des sorties de `runs/<run>/` (JSON, Markdown, figures) pour les runs discutés dans
`docs/MS-VICRegL_synthese_SNIP_pooling.docx`. Les checkpoints (`ckpt.pt`, ~30 Mo chacun)
et les logs d'entraînement bruts restent dans `runs/` (gitignoré) — pas nécessaires pour
relire les chiffres, régénérables via les commandes `scripts/run_all.py` /
`scripts/04_compare.py` / `scripts/08_holdout_D.py` documentées dans le README principal.

- `pretrain/` — checkpoint de référence (B∪C, sans SNIP). `RESULTS.md`/`comparison.json`
  régénérés avec le pooling spatial (`n_segments="max"`) devenu le défaut ; `amr.json`/
  `finegrain.json` restent sur l'ancien pooling (`n_segments=1`, épinglé pour ces
  benchmarks à petits effectifs).
- `pretrain_snip/` — SNIP, gamma=20 (confondu avec le correctif gamma, voir synthèse §1).
- `pretrain_snip_gamma8/` et `pretrain_snip_gamma8_seed1/` — SNIP, gamma=8 isolé,
  seed=0 et seed=1 (réplication).
- `holdout_D*/` — le même éventail de checkpoints évalués sur le centre D tenu à l'écart
  (le benchmark phare (B+C)→D).
- `summary_this_session.json` — extraction condensée des chiffres clés (balanced accuracy,
  accuracy, F1-macro, MCC, test de McNemar) utilisée pour rédiger la synthèse.

Voir `docs/MS-VICRegL_synthese_SNIP_pooling.docx` pour l'interprétation et les réserves
(confusion gamma/SNIP, variance inter-graines) — ces fichiers sont la donnée brute, pas
l'analyse.
