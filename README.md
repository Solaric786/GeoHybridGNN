# GeoHybridGNN

Geology-informed deep learning workflow for porphyry copper prospectivity mapping in the Western Chagai Belt, Pakistan.

## Overview

This repository provides the public-facing code and documentation snapshot for the **GeoHybridGNN** project. The study focuses on integrating geospatial preprocessing and geology-informed machine learning for porphyry copper prospectivity analysis in the Western Chagai Belt.

The current public version is a **clean release snapshot**, not yet a complete end-to-end reproduction package of the full local research workspace.

## Current repository contents

- `public_code_snapshot/` — publicly prepared code snapshot from the working project, including visible preprocessing and Google Earth Engine scripts.
- `docs/` — supporting release notes and repository-planning documents.
- `requirements-public.txt` — conservative public environment file based on visible imports.
- `CITATION.cff` — citation metadata for the repository.
- `.zenodo.json` — metadata for future Zenodo archiving.
- `CODE_AND_DATA_AVAILABILITY.md` — suggested wording for manuscript code/data availability statements.

## Repository structure

```text
GeoHybridGNN/
├── docs/
├── public_code_snapshot/
├── .gitignore
├── .zenodo.json
├── CITATION.cff
├── CODE_AND_DATA_AVAILABILITY.md
├── README.md
└── requirements-public.txt
````

## Scope of this public release

This repository is intended to support:

* public documentation of the project structure,
* release of selected preprocessing and workflow scripts,
* citation and archival of the public code snapshot,
* future expansion into a more complete reproducibility package.

At this stage, the repository should be understood as a **carefully prepared public release scaffold**.

## Important limitation

The original working project included materials that were not all directly extractable in this environment. As a result, this public repository currently reflects the files that were accessible and suitable for release preparation, rather than the full internal project tree.

Accordingly, this version should not be interpreted as the final complete research archive.

## What is not included

This public repository does **not** include:

* raw proprietary or restricted data,
* heavy intermediate rasters and large generated outputs,
* local machine paths or private environment details,
* sensitive or internal-only working notes,
* unreleased full project artifacts that were not safely verified for publication.

## Reproducibility note

The repository currently contains a public code snapshot and release documentation. Additional workflow components, cleaned training scripts, and reproducibility materials may be added in later versions after final verification for public release.

## Related manuscript context

This repository supports the GeoHybridGNN study on geology-informed deep learning for porphyry copper prospectivity mapping in the Western Chagai Belt. The public code and metadata here are intended to accompany manuscript submission, review, and future archival release.

## Citation

Please use the repository citation metadata in `CITATION.cff`. A Zenodo-linked citation can be added after the first archived release.

## Contact and updates

Future cleaned releases may expand the repository with additional scripts, workflow clarification, and archival metadata.

