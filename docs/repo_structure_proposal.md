# Proposed clean repository structure

```text
geohybridgnn-chagai-porphyry-prospectivity/
├── README.md
├── LICENSE                # choose before publishing
├── CITATION.cff
├── .zenodo.json
├── .gitignore
├── requirements-public.txt
├── environment.yml        # optional
├── data/
│   ├── README.md          # explain what is public vs unavailable
│   ├── sample/            # tiny example inputs only
│   └── external/          # ignored; user-supplied data live here
├── src/
│   ├── preprocess/
│   ├── digitize/
│   ├── graph/
│   ├── models/
│   ├── evaluation/
│   └── visualization/
├── notebooks/             # optional light demos only
├── outputs/               # gitignored
├── figures/
│   └── README.md
└── docs/
    ├── manuscript_support/
    └── reproducibility.md
```

## Recommended cleanup mapping

### Keep as active scripts
- preprocess: `00`–`09` and QC scripts
- graph construction: `10_build_graph_gridcells_v1.py`, `12_export_graph_to_pyg.py`, `15_make_node_labels_and_blockcv.py`
- model training: `21_train_cnn_patch_encoder_v1.py`, `22_apply_cnn_encoder_to_nodes_v1.py`, `23_train_hybrid_gnn_graphsage_v1.py`
- evaluation: `27_eval_spatial_block_cv_from_ensemble_maps.py`, `28_baseline_logreg_spatial_block_cv.py`, metric/ROC scripts
- figures: the final publication map scripts only

### Move to `legacy/` or exclude
- multiple repeated figure scripts (`24_make_usgs_paper_maps_v1` ... `v10`)
- superseded evaluation variants
- scratch tests (`test.py`)
- files with broken names like `export_cu_chagai_to_parquet.py .py`

## High-priority code cleanup before public release
- Replace all hard-coded local paths like `D:\Atif's_Science\...` with relative paths or config arguments.
- Move all user settings to one config file or command-line interface.
- Add short docstrings and usage examples to top-level scripts.
- Make the final workflow order explicit in the README.
