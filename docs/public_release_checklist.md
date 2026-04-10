# Public release checklist

## Must-fix before GitHub
- [ ] Remove hard-coded local Windows paths.
- [ ] Remove or ignore generated rasters, model checkpoints, large outputs, and temporary files.
- [ ] Choose one canonical script for each pipeline stage.
- [ ] Rename unclear files and remove duplicate publication-map versions.
- [ ] Add a README with project purpose, workflow, and reproduction steps.
- [ ] Add dependency file (`requirements-public.txt` and/or `environment.yml`).
- [ ] Add citation metadata (`CITATION.cff`).
- [ ] Decide on a license.
- [ ] Add a data availability note explaining what data cannot be redistributed.

## Strongly recommended
- [ ] Add a tiny sample dataset or a mock example to test the pipeline shape.
- [ ] Add `--help` or config-based execution to main scripts.
- [ ] Add repository topics on GitHub: `remote-sensing`, `mineral-prospectivity`, `graph-neural-networks`, `geoscience`, `python`.
- [ ] Create a release tag like `v1.0.0` once the manuscript-support version is frozen.

## Zenodo release path
- [ ] Push cleaned repository to GitHub.
- [ ] Log in to Zenodo with GitHub.
- [ ] Enable the repository in Zenodo.
- [ ] Add `.zenodo.json` and `CITATION.cff` to the repo.
- [ ] Create a GitHub release.
- [ ] Let Zenodo archive that release and mint the DOI.
