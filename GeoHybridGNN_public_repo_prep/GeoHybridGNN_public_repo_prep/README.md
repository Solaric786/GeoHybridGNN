# GeoHybridGNN: GitHub-ready public release prep

This folder is a **public-release preparation package** for the GeoHybridGNN project used for porphyry copper prospectivity mapping in the Western Chagai Belt, Pakistan.

It is designed to help convert a working local research project into a cleaner **GitHub repository** and, if desired, a **Zenodo-archived release**.

## What is included here

- `public_code_snapshot/` — the preprocessing and GEE scripts that were directly accessible in this workspace.
- `docs/archive_manifest_from_rar_listing.txt` — a manifest of additional files detected inside the uploaded `02_Code.rar` archive.
- `docs/repo_structure_proposal.md` — a recommended clean repository structure.
- `docs/public_release_checklist.md` — a step-by-step checklist before pushing anything public.
- `requirements-public.txt` — a conservative public environment file based on visible imports.
- `CITATION.cff` and `.zenodo.json` — starter metadata files for citation and Zenodo archiving.
- `CODE_AND_DATA_AVAILABILITY.md` — suggested wording for manuscript availability statements.

## Important limitation

The uploaded project archive was in **RAR** format. In this environment, the archive contents were readable as a **file list**, but the full files inside the archive could not be extracted automatically because no working RAR extraction binary was available.

Accordingly, this package is a **careful best-effort public-release scaffold**, not a final one-click export of the entire project.

## Recommended public repository name

`geohybridgn-chagai-porphyry-prospectivity`

## Suggested first public release contents

Public release should include:
- preprocessing scripts
- graph-building and model-training scripts after cleanup
- README with workflow overview
- environment file
- figure-generation scripts that reproduce the manuscript maps
- metadata files (`CITATION.cff`, `.zenodo.json`)

Public release should **not** include:
- raw proprietary data
- intermediate rasters and heavy generated outputs
- private local paths
- personal emails or internal notes beyond the manuscript authorship metadata
- accidental duplicates / version clutter (`v1`, `v2`, ..., `v10`) unless clearly archived in a legacy folder

## Next best action

1. Convert the RAR archive to ZIP locally and re-upload it, or upload the key `src/graph/` scripts directly.
2. Use the structure proposed in `docs/repo_structure_proposal.md`.
3. After the repo is pushed to GitHub, connect it to Zenodo and archive the first release.
