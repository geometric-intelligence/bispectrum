# Release process

Release artifacts are published to [PyPI](https://pypi.org/project/bispectrum/)
as a source distribution + universal wheel built by `hatchling`. Only files
under `src/bispectrum/` are packaged; `experiments/`, `benchmarks/`, and `tests/`
are excluded.

The release pipeline is **automated** via GitHub Actions
(`.github/workflows/publish.yml`). Pushing a tag matching `v*` triggers a
build that:

1. Verifies the tag matches `[project].version` in `pyproject.toml`.
2. Builds the wheel + sdist with `uv build`.
3. Runs `twine check` for metadata sanity.
4. Uploads to PyPI via [Trusted Publishing][trusted] (no long-lived API token).
5. Creates a GitHub Release with the artifacts attached and auto-generated
   release notes.

## One-time setup (per-project, done once ever)

Configure PyPI as a "trusted publisher" for this repo so GitHub Actions can
upload without a stored API token:

1. Go to <https://pypi.org/manage/account/publishing/> and add a new
   **pending publisher** (use this if `bispectrum` doesn't exist on PyPI yet)
   or click "Add a new publisher" on the project page (if it already exists).
2. Fill in:
   - PyPI Project Name: `bispectrum`
   - Owner: `geometric-intelligence`
   - Repository: `bispectrum`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In GitHub, create a deployment environment named `pypi`:
   `Settings → Environments → New environment → "pypi"`. (You can also
   add a "required reviewer" rule here so each release needs a manual
   approval click before Actions uploads.)

Done. No tokens, no `~/.pypirc`, nothing to rotate.

## Cutting a release

The version lives in **one place**: `[project].version` in `pyproject.toml`.
`bispectrum.__version__` is read at runtime via `importlib.metadata`, so it
stays in sync automatically. Direct pushes to `main` are not allowed; the
version bump goes through a PR so CI runs and there's an audit trail.

Set the target version once:

```bash
export VERSION=0.3.2
```

### Step 1 — Run the test suite locally

```bash
uv sync --extra dev
uv run pytest tests/ -n auto -q
```

Don't tag a release on top of red tests. If anything fails, fix it first.

### Step 2 — Open a release PR

```bash
git checkout main && git pull --ff-only origin main
git checkout -b "release/v$VERSION"

sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
grep -m1 '^version' pyproject.toml   # sanity check

git add pyproject.toml
git commit -m "Release v$VERSION"
git push -u origin HEAD

gh pr create --base main --title "Release v$VERSION" \
    --body "Bump version to v$VERSION for PyPI release."
```

### Step 3 — Wait for CI, then merge

```bash
gh pr checks --watch                  # block until CI is green
gh pr merge --squash --delete-branch  # or --merge if you prefer no squash
```

### Step 4 — Tag and push

This is the one step that triggers everything:

```bash
git checkout main && git pull --ff-only origin main
git tag -a "v$VERSION" -m "v$VERSION"
git push origin "v$VERSION"
```

The `Publish to PyPI` workflow now runs and:

- Builds + uploads to PyPI.
- Creates a GitHub Release with the wheel + sdist attached and auto-generated
  notes from PRs since the previous tag.

Watch progress:

```bash
gh run watch                                              # latest run
gh run view --log                                         # full logs
gh release view "v$VERSION"                               # the release
```

PyPI rejects re-uploads of the same version, so always bump first.

## Verifying a release

```bash
pip index versions bispectrum                                       # did it land?
uv run --isolated --no-project --with "bispectrum==$VERSION" \
    --python 3.12 python -c "import bispectrum; print(bispectrum.__version__)"
```

## Inspecting the artifacts (debugging only)

To inspect what *would* be uploaded without triggering Actions:

```bash
rm -rf dist/ && uv build
unzip -l dist/bispectrum-*.whl       # should be only bispectrum/...
tar -tzf dist/bispectrum-*.tar.gz    # src/bispectrum/... + LICENSE/README/pyproject
```

If `paper/` or `benchmarks/` show up there, the include rules in
`[tool.hatch.build]` got broken — fix before tagging.

## Supported runtime matrix

- **Python**: 3.12 only (`torch-harmonics` 0.9.0 ships no cp313 wheel).
- **PyTorch**: `>=2.10`. The `torch-harmonics` 0.9.0 PyPI wheel links against
  `c10::TensorImpl::decref_pyobject`, a symbol that first appeared in torch
  2.10. Older torch causes an `ImportError` at `import torch_harmonics`.
- **CUDA**: end users on driver `<12.8` need to override the torch wheel:
  ```bash
  pip install bispectrum --extra-index-url https://download.pytorch.org/whl/cu128
  ```
  (The `[tool.uv.*]` blocks in `pyproject.toml` only affect local development
  via `uv sync`/`uv run` — they are ignored by `pip` and `hatch`.)

## When `torch-harmonics` ships a new release

Their declared `requires_python` and `requires_dist` constraints are
unreliable (the wheels' actual ABI requirements are tighter than what they
advertise). Re-test before bumping our floor:

```bash
curl -s https://pypi.org/pypi/torch-harmonics/json \
  | python -c 'import json,sys; r=json.load(sys.stdin)["releases"]; latest=sorted(r)[-1]; print(latest, [f["filename"] for f in r[latest]])'
```

Check that the latest wheels have `cp313` (or whatever Python you want to
support) and run the smoke test below against the lowest torch you intend
to advertise.

## Smoke test (run before tagging)

Pins to the lowest torch / torch-harmonics we advertise to catch ABI drift.
Build first if you haven't (`uv build`):

```bash
export TORCH_FLOOR=2.10.0
export TH_FLOOR=0.9.0

uv run --isolated --no-project \
    --with "dist/bispectrum-$VERSION-py3-none-any.whl" \
    --with "torch==$TORCH_FLOOR" \
    --with "torch-harmonics==$TH_FLOOR" \
    --python 3.12 \
    python -c "
import torch
from bispectrum import CnonCn, DnonDn, SO2onS1, TorusOnTorus, SO2onDisk, SO3onS2, OctaonOcta
torch.manual_seed(0)
assert CnonCn(n=8)(torch.randn(2, 8)).shape == (2, 8)
assert DnonDn(n=8)(torch.randn(2, 16)).shape[0] == 2
assert SO3onS2(lmax=4, nlat=32, nlon=64)(torch.randn(1, 32, 64)).shape[0] == 1
assert OctaonOcta()(torch.randn(2, 24)).shape == (2, 172)
print('OK')
"
```

## Manual fallback (Actions is down, etc.)

If for some reason the workflow can't run, you can still publish from your
laptop. You'll need a PyPI API token from
<https://pypi.org/manage/account/token/>:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEI...
rm -rf dist/ && uv build && uvx twine check dist/* && uvx twine upload dist/*
```

Then create the GitHub Release manually:

```bash
gh release create "v$VERSION" dist/* --title "v$VERSION" --generate-notes
```

## Dry-run on TestPyPI (optional)

If you want to validate the flow end-to-end without touching real PyPI,
configure a second trusted publisher pointing at <https://test.pypi.org>
and a `testpypi` environment, then add a parallel workflow that triggers
on a different tag pattern (e.g. `testv*`). Or just run the manual fallback
against `--repository testpypi`:

```bash
uvx twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ bispectrum
```

The `--extra-index-url` is required because TestPyPI does not mirror
`torch`/`numpy`.

[trusted]: https://docs.pypi.org/trusted-publishers/
