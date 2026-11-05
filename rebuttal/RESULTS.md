# Results store

Run outputs are not in git: `results.json`, training logs, best-model checkpoints, analysis tables, device timings and the job-runner state. They live in `results/` at the repository root, which is gitignored, at the same relative path they had in the tree. Each original location is a relative symlink into the store, so sweep, analysis and figure scripts read and write the same paths as before. `results` can itself be a symlink to a bigger disk.

| Location (symlink)                      | Store path                                      |
| --------------------------------------- | ----------------------------------------------- |
| `experiments/<dataset>/*_results*`      | `results/experiments/<dataset>/*_results*`      |
| `rebuttal/results`                      | `results/rebuttal/results`                      |
| `benchmarks/figures/paper_timings.json` | `results/benchmarks/figures/paper_timings.json` |

`rebuttal/results_manifest.tsv` (in git) lists the path, size and sha256 of every stored file. At the camera-ready freeze the store held 3,409 files, 3.39 GiB in total:

| Kind                                  | Files |     Size |
| ------------------------------------- | ----: | -------: |
| Checkpoints (`*.pt`)                  |   928 | 3.34 GiB |
| `results.json` and other JSON         | 1,880 |   31 MiB |
| Training logs                         |   555 |   19 MiB |
| Tables, figures, environment captures |    46 |    2 MiB |

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

**Plan: a Hugging Face dataset, `geometric-intelligence/bispectrum-results`, private until the camera-ready is public.** It is versioned like a git repo, has no LFS quota tied to the code repository, serves partial downloads by glob, and is browsable. Two tiers by pattern:

- **Light (about 51 MiB):** everything except `*.pt`. This is enough to regenerate every figure, table and significance test.
- **Full (3.4 GiB):** adds the checkpoints, needed only to re-evaluate trained models.

```bash
pip install -U huggingface_hub            # provides the `hf` CLI
hf auth login
hf repo create geometric-intelligence/bispectrum-results --repo-type dataset --private
hf upload-large-folder geometric-intelligence/bispectrum-results results --repo-type dataset

# On a fresh clone: the light tier, then links and a check
hf download geometric-intelligence/bispectrum-results --repo-type dataset --local-dir results --exclude '*.pt'
python rebuttal/results_store.py link
python rebuttal/results_store.py verify --include '*.json' '*.log'
```

At the camera-ready freeze:

1. Tag the dataset revision that matches the paper's code commit, and record both hashes here.
2. Deposit the light tier on Zenodo for a DOI, and cite it in the paper's reproducibility statement. Checkpoints stay on the Hugging Face dataset.
3. Make the dataset public.

**Rejected alternatives:**

- **Git LFS on GitHub:** the checkpoints alone exceed the free storage and bandwidth quota, and every clone of the library pays for them.
- **GitHub release assets:** no browsing, partial fetch or per-file versioning.
- **S3 or R2:** needs credential management and has no public browsing.

`hf download --local-dir` and `hf upload-large-folder` keep their state in `results/.cache/`; `results_store.py` ignores dot-directories.
