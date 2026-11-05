# Results store

Run outputs are not in git: `results.json`, training logs, best-model checkpoints, analysis tables, device timings and the job-runner state. They live in `results/` at the repository root, which is gitignored, at the same relative path they had in the tree. Each original location is a relative symlink into the store, so sweep, analysis and figure scripts read and write the same paths as before. `results` can itself be a symlink to a bigger disk.

| Location (symlink)                      | Store path                                      |
| --------------------------------------- | ----------------------------------------------- |
| `experiments/<dataset>/*_results*`      | `results/experiments/<dataset>/*_results*`      |
| `rebuttal/results`                      | `results/rebuttal/results`                      |
| `benchmarks/figures/paper_timings.json` | `results/benchmarks/figures/paper_timings.json` |

`rebuttal/results_manifest.tsv` (in git) lists the path, size and sha256 of every stored file. At the camera-ready freeze the store held 3,446 files, 3.39 GiB in total:

| Kind                                                | Files |     Size |
| --------------------------------------------------- | ----: | -------: |
| Checkpoints and replot caches (`*.pt`)              |   928 | 3.34 GiB |
| `results.json` and other JSON (`*.json`, `*.jsonl`) | 1,886 |   32 MiB |
| Training and solver logs                            |   589 |   19 MiB |
| Tables, figures, environment captures, dataset card |    43 |    2 MiB |

## Commands

```bash
python rebuttal/results_store.py adopt      # move real run-output dirs into results/, link back
python rebuttal/results_store.py link       # recreate the symlinks from results/ (fresh clone)
python rebuttal/results_store.py manifest   # rewrite rebuttal/results_manifest.tsv
python rebuttal/results_store.py verify     # compare results/ with the manifest (sha256)
```

New sweeps, and `archive_stale_results.py`, create real directories at the old paths. Run `adopt` afterwards, then `manifest`, and commit the manifest.

## Existing clones with results tracked in place (bongo, l8a100, H100)

The commit that introduced the store untracks every result file, so a plain `git pull` deletes the tracked `results.json` files from the working tree (untracked checkpoints and logs survive). Move them into the store first:

```bash
git fetch origin
git show origin/johmathe/camera-ready-final-results:rebuttal/results_store.py > /tmp/results_store.py
python /tmp/results_store.py --repo . adopt
git pull
```

## Where the store is published

**Hugging Face dataset: [`geometric-intelligence/bispectrum`](https://huggingface.co/datasets/geometric-intelligence/bispectrum).** It is versioned like a git repo, serves partial downloads by glob, and is browsable. The dataset card is `results/README.md`.

**Only JSON is published** (`*.json`, `*.jsonl`: about 1,900 files, 32 MiB). Every figure, table and significance test is computed from these. Checkpoints, training logs, derived tables and environment captures stay in the local stores on the machines that produced them, and the manifest records their hashes.

Published revision: [`95a2b26`](https://huggingface.co/datasets/geometric-intelligence/bispectrum/commit/95a2b26fc953a2ec46a8ba771147f59e8aaf4198) (2026-11-05), matching this manifest. `results-json.tar.gz` there has sha256 `b95ca34dec0af671c06fe68a49d6a0a556dc5b335f5cd64f3094bf5fdd33140e`. Downloaded into a fresh clone, it passes `verify` and rebuilds `make_grid_figure.py`'s outputs byte for byte, apart from PDF creation dates.

`hf` is the Hugging Face CLI (`pip install -U huggingface_hub`, or run it as `uvx --from huggingface_hub hf`). The token needs write access to the dataset.

The dataset carries the JSON twice: as loose files for browsing and partial fetches, and as `results-json.tar.gz` (6.5 MiB) for a one-request download. The archive is built with `pack`, which fixes owners, permissions and the gzip header, so an unchanged store packs to the same bytes.

```bash
# Publish or update: the loose files, then the archive
hf upload geometric-intelligence/bispectrum results . --repo-type dataset \
  --include '*.json' --include '*.jsonl' --include README.md --commit-message "..."
python rebuttal/results_store.py pack /tmp/results-json.tar.gz --include '*.json' '*.jsonl'
hf upload geometric-intelligence/bispectrum /tmp/results-json.tar.gz results-json.tar.gz --repo-type dataset

# On a fresh clone: fetch, link, check
curl -L https://huggingface.co/datasets/geometric-intelligence/bispectrum/resolve/main/results-json.tar.gz | tar -xz
python rebuttal/results_store.py link
python rebuttal/results_store.py verify --include '*.json' '*.jsonl'
```

`hf download geometric-intelligence/bispectrum --repo-type dataset --local-dir results --exclude '*.tar.gz'` fetches the loose files instead, and accepts `--include` for one experiment. Anonymously, its roughly 1,900 requests hit Hugging Face's rate limit; the CLI waits about four minutes and retries. Running `hf auth login` first raises the limit.

At the camera-ready freeze:

1. Tag the dataset revision that matches the paper's code commit, and record both hashes here. Point the dataset card's `git clone -b` and `resolve/main` URLs at the tags.
2. Deposit `results-json.tar.gz` on Zenodo for a DOI, and cite it in the paper's reproducibility statement.

**Rejected alternatives:**

- **Git LFS on GitHub:** every clone of the library would pay for the results.
- **GitHub release assets:** no browsing, partial fetch or per-file versioning.
- **S3 or R2:** needs credential management and has no public browsing.

`hf download --local-dir` keeps its state in `results/.cache/`, and the dataset repo carries a `.gitattributes`; `results_store.py` ignores dot-entries.
