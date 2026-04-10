# Quick technical review for public release

## What looks good already
- The project has a clear staged workflow: preprocessing -> graph construction -> CNN patch encoding -> hybrid GraphSAGE training -> evaluation -> publication figures.
- Script naming suggests a reproducible paper pipeline.
- The manuscript already documents the final modeling choices clearly.

## Main risks before making the repository public
1. **Hard-coded local paths** appear in the visible preprocessing scripts.
2. **Version clutter** in the graph/figure scripts will confuse outside users.
3. **Data redistribution limits** need to be stated clearly because not all inputs should be uploaded.
4. **RAR archive limitation** prevented full automated extraction in this environment, so final cleanup of graph scripts still needs one more pass after the archive is re-uploaded as ZIP or the `src/graph/` files are shared directly.

## Best publication strategy
- Create the GitHub repository first.
- Push a cleaned manuscript-support version.
- Connect the repository to Zenodo.
- Create the first GitHub release only when the code snapshot is frozen.
