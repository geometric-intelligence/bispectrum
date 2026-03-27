# Review of `paper.tex`

This review is intentionally harsh and reviewer-oriented. It focuses on what is currently unsupported, overstated, internally inconsistent, or likely to trigger serious pushback from a careful NeurIPS reviewer. It is based on `paper/paper.tex`, the package code under `src/bispectrum/`, the experiment code and artifacts under `paper/experiments/pcam/`, the benchmark code, and the literature/planning notes under `docs/`.

## Top-line assessment

The repository has real substance. The library itself is technically interesting, the octahedral inversion work looks real and nontrivial, and there is enough code to support a serious software paper.

The paper, however, is currently much weaker than the repo. In its present form it reads like an ambitious submission outline with some strong ingredients, not like a finished research paper. The most serious problem is not lack of quality in the code; it is the mismatch between the strength of the claims and the level of support actually present in the manuscript and experimental artifacts.

## Findings

### 1. Critical: the manuscript is still a scaffold, not a reviewable paper

The paper contains TODOs and placeholders in the abstract, introduction, preliminaries, theorem statement, benchmarks, experiments, conclusion, roadmap, acknowledgments, and appendices. At the same time, the surrounding prose already speaks as if the scientific and empirical case is complete.

This is the single biggest issue. A reviewer will not interpret this as "some rough edges remain"; they will interpret it as "core parts of the submission are missing."

Relevant examples from `paper/paper.tex`:

- The abstract is still a `\todo` block with bullets instead of text.
- The introduction opens with a `\todo`.
- The steerable-to-Fourier theorem is completely unstated.
- The benchmark figures are placeholders.
- The main result tables still contain `X.XXX` placeholders.
- The appendix sections that should carry proof, convergence, and benchmark detail are almost entirely TODOs.

### 2. Critical: the current PCam artifacts do not support the main empirical framing

The paper claims a five-model head-to-head comparison on PatchCamelyon with a shared backbone and three seeds per model. The saved results currently do not match that story.

From `paper/experiments/pcam/pcam_results/` and `pcam_results_1pct/`:

- `standard` has three seeds at `train_fraction = 1.0`
- `norm` has two seeds at `train_fraction = 1.0`
- `gate`, `fourier_elu`, and `bispectrum` only appear for seed 42
- those `gate`, `fourier_elu`, and `bispectrum` runs are at `train_fraction = 0.01`, not `1.0`

This means the currently saved artifacts are not a fair main-table comparison.

Concrete examples:

- `paper/experiments/pcam/pcam_results/standard_c8_seed42/results.json`
- `paper/experiments/pcam/pcam_results/standard_c8_seed123/results.json`
- `paper/experiments/pcam/pcam_results/standard_c8_seed456/results.json`
- `paper/experiments/pcam/pcam_results/norm_c8_seed42/results.json`
- `paper/experiments/pcam/pcam_results/norm_c8_seed123/results.json`
- `paper/experiments/pcam/pcam_results/bispectrum_c8_seed42/results.json`
- `paper/experiments/pcam/pcam_results/fourier_elu_c8_seed42/results.json`
- `paper/experiments/pcam/pcam_results/gate_c8_seed42/results.json`

The most dangerous consequence is that the paper currently implies a result that the checked-in artifacts do not actually support.

### 3. Critical: "same backbone, only pooling differs" is false in the current experimental implementation

The manuscript says the five models share the same backbone and differ only in the invariant pooling layer. The implementation in `paper/experiments/pcam/model.py` does not support that claim in a strict sense.

Problems:

- the gated model increases channels through `gate_factor = 2`
- the bispectrum model expands features substantially before a `1x1` projection
- the resulting parameter counts differ dramatically across models

Saved parameter counts in the current results:

- `standard`: `101,959`
- `norm`: `790,717`
- `bispectrum`: `919,609`
- `gate`: `1,576,159`

This is not a cosmetic discrepancy. It is a serious experimental confound. If a reviewer notices this, they can reasonably argue that any gains may be due to capacity differences rather than invariance quality or information preservation.

Relevant files:

- `paper/experiments/pcam/model.py`
- `paper/experiments/pcam/pcam_results/*/results.json`

### 4. High: the theory section overstates what is actually complete

The internal theory notes are more careful than the paper draft. In particular:

- the completeness guarantees apply to the proper bispectrum of the relevant spectral object
- for steerable CNN outputs, completeness is not automatically a statement about the original image
- the learned steerable coefficients are not automatically the same thing as canonical harmonic-analysis Fourier coefficients

Your own note in `docs/per_pixel_steerable_bispectrum.tex` states the scope correctly: completeness applies to the per-pixel fiber signal, not directly to the original image. Your `docs/steerable_bispectrum_notes.md` goes further and explicitly warns that the usual bispectrum completeness/invertibility guarantees do not automatically transfer to learned steerable responses.

The paper draft softens or drops those caveats when it matters most:

- the abstract
- the comparison table
- the contribution list
- the experimental motivation for "preserving all discriminative information"

This creates a serious risk of overclaiming.

Relevant files:

- `paper/paper.tex`
- `docs/per_pixel_steerable_bispectrum.tex`
- `docs/steerable_bispectrum_notes.md`

### 5. High: the steerable-to-Fourier bridge is presented as solved, but the manuscript still has not stated or proved it

This is one of the core scientific contributions the paper wants to claim. But in the actual LaTeX:

- the theorem statement is still a TODO
- the appendix proof section is still a TODO
- the discussion speaks as if the problem has already been fully resolved

This creates an internal contradiction between the rhetoric of the paper and the actual content on the page.

Your planning document `docs/neurips_steerable_bispectrum_plan.md` is still realistic about this risk. It explicitly marks the key theory path as open or partial. The draft paper is much more confident than the supporting material warrants.

Relevant files:

- `paper/paper.tex`
- `docs/neurips_steerable_bispectrum_plan.md`

### 6. High: the API and design claims in the paper are broader than what the package implements

The package is real and useful, but the paper overstates its uniformity.

#### `selective=False` is not generally available

The design principles section implies that all modules are selective by default and that the full bispectrum is available through `selective=False` for debugging/comparison. That is not true across the library.

Examples:

- `src/bispectrum/cn_on_cn.py`: full mode exists
- `src/bispectrum/torus_on_torus.py`: full mode exists
- `src/bispectrum/dn_on_dn.py`: full mode raises `NotImplementedError`
- `src/bispectrum/octa_on_octa.py`: full mode raises `NotImplementedError`
- `src/bispectrum/so2_on_disk.py`: full mode raises `NotImplementedError`
- `src/bispectrum/so3_on_s2.py`: selective mode is reserved for future use; selective SO(3) is explicitly open

#### The "uniform API contract" is too strong

The paper says every module returns a complex tensor and that `index_map` maps to an irrep pair.

That is not uniformly true:

- `DnonDn` returns a real bispectrum
- `SO3onS2.index_map` uses triples `(l1, l2, l)`
- other modules use different tuple structures

These are fixable wording issues, but if left as-is they invite easy reviewer nitpicks.

Relevant files:

- `paper/paper.tex`
- `src/bispectrum/cn_on_cn.py`
- `src/bispectrum/dn_on_dn.py`
- `src/bispectrum/octa_on_octa.py`
- `src/bispectrum/so2_on_disk.py`
- `src/bispectrum/so3_on_s2.py`
- `src/bispectrum/torus_on_torus.py`

### 7. High: the benchmark script cited by the paper appears broken

The paper points to `benchmarks/benchmark.py --paper`. But `benchmarks/benchmark.py` imports `SO2onD2`, while the package currently exports `SO2onDisk`.

That means a reviewer trying to run the benchmark script could hit an import error immediately.

This also reveals naming drift in the repo between `SO2onD2` and `SO2onDisk`, including in `docs/`.

Relevant files:

- `benchmarks/benchmark.py`
- `src/bispectrum/__init__.py`
- `docs/steerable_bispectrum_notes.md`
- `docs/so2_on_d2_implementation_notes.md`

### 8. High: the RotMNIST contrastive story looks unsupported in the current tree

The abstract, contribution list, and experiments section all position the RotMNIST bispectral contrastive regularizer as a major empirical result.

I could find the drafted section in `paper/paper.tex`, but I did not find matching experiment code or result artifacts in the repository. Right now, that makes the RotMNIST section read like a proposed experiment rather than a completed one.

If this experiment exists elsewhere and is not yet committed, that is fine. But in the current repo snapshot, it is not supported.

Relevant files:

- `paper/paper.tex`
- `paper/experiment_candidates.md`

### 9. Medium: some factual claims are simply wrong or not evidenced in-tree

#### "Minimal dependencies: only torch and numpy"

This is false. The package depends on `torch-harmonics`.

Relevant files:

- `paper/paper.tex`
- `pyproject.toml`
- `src/bispectrum/so3_on_s2.py`

#### "`>90%` test coverage"

This may be true on Codecov, but it is not substantiated in the repo snapshot itself. The CI runs coverage and uploads it, but the checked-in evidence I saw is a badge, not a durable number.

Relevant files:

- `paper/paper.tex`
- `README.md`
- `.github/workflows/tests.yml`

#### "First PyTorch library"

This may well be true in practice, but as written it is a global novelty claim. Unless you have done an exhaustive software search, it should probably be scoped as "to our knowledge" or narrowed.

### 10. Medium: the literature positioning is directionally good but underplays the strongest prior line

The current draft cites Oreiller 2022, but your own notes make clear that the more complete prior line is:

- Andrearczyk et al. 2019
- Oreiller et al. 2020
- Oreiller et al. 2022

That matters because the novelty question is not "has anyone ever put bispectrum-like invariants into CNNs?" The harder and more honest novelty claim is closer to:

- nobody has provided a practical selective multi-group PyTorch library of this scope
- nobody has resolved the non-abelian steerable/Fourier mismatch in the form you want
- nobody has dealt with the octahedral inversion issue in software

That is still a strong story. But it is more defensible than implying the broader bispectral-CNN direction is new.

Relevant files:

- `paper/paper.tex`
- `docs/steerable_bispectrum_notes.md`
- `docs/sanbord_miolane_2025/sections/lit-review.tex`

### 11. Medium: the paper is trying to be too many paper types at once

At the moment the draft tries to simultaneously be:

- a software paper
- a theory paper
- a benchmarking paper
- an equivariant CNN empirical paper
- a representation-theory bridge paper
- a contrastive learning paper

That breadth would already be hard to pull off in a polished manuscript. In the current state, it dilutes the strongest contributions.

The strongest scientifically credible core, based on the repo, is:

1. a serious PyTorch bispectrum library across several groups/domains
2. a real octahedral inversion failure and fix
3. a careful treatment of the steerable-to-Fourier gap, with exact scope

Everything beyond that should be included only if it is actually complete and defensible.

## Questions I would expect from reviewers

These are the sorts of questions the current draft invites:

- Is the steerable-to-Fourier bridge actually proved, or only conjectured / partially characterized?
- Are the empirical gains due to better invariants, or simply larger parameter count?
- Are the PCam comparisons apples-to-apples if some models only have 1% data runs saved?
- Why should the reader trust claims about RotMNIST if the code/results are not visible?
- In what exact sense is the invariant "complete" when used inside learned steerable networks?
- Is the benchmark code actually runnable from a clean install?
- How much of the paper is a software contribution vs a scientific contribution?

## What is already strong

This should not get lost in the critique.

### The library looks real

The package under `src/bispectrum/` is substantial. This is not a toy wrapper. It includes actual implementations for:

- `CnonCn`
- `DnonDn`
- `TorusOnTorus`
- `SO2onS1`
- `SO2onDisk`
- `SO3onS2`
- `OctaonOcta`

### The octahedral inversion story is genuinely interesting

Among all the paper's claims, this is one of the most convincing and distinctive. The code and notes support a real phenomenon:

- the selective octahedral bispectrum is fine
- the naive bootstrap inversion fails because the continuous `SO(3)` ambiguity is not absorbed by the octahedral CG structure
- a bootstrap + LM + multi-start correction is implemented

This is exactly the kind of "nontrivial engineering plus mathematical insight" that can anchor a strong software paper.

Relevant files:

- `src/bispectrum/octa_on_octa.py`
- `docs/inversion_gap_for_octahedral_group.md`

### The package/repo quality is credible

The repo supports a real software story:

- packaging via `pyproject.toml`
- installable package
- tests
- CI
- coverage upload
- MIT license

That part of the paper can be made credible with relatively little effort.

## Best scientific framing available from the current repo

If I were optimizing for acceptance probability rather than maximal ambition, I would frame the contribution more tightly around:

1. `bispectrum` is a practical PyTorch library implementing selective bispectra across multiple groups/domains.
2. A key scientific contribution is the octahedral inversion failure analysis and correction.
3. A second scientific contribution is a scoped and carefully stated bridge from steerable outputs to valid bispectral computation, with explicit assumptions and limitations.
4. The experiments should only claim what the saved artifacts actually support.

That is still a strong paper.

## Recommendation

The fastest path to a strong submission is probably not to add more prose. It is to narrow the claims until every major sentence is directly defensible from:

- the code
- the experiments
- the theorem statements actually written in the paper

Right now the repo is ahead of the manuscript. The paper needs to catch up by becoming more precise, more scoped, and more honest about what is complete versus what is still planned.

## File references used in this review

- `paper/paper.tex`
- `paper/references.bib`
- `paper/experiments/pcam/README.md`
- `paper/experiments/pcam/model.py`
- `paper/experiments/pcam/train.py`
- `paper/experiments/pcam/pcam_results/*/results.json`
- `paper/experiments/pcam/pcam_results_1pct/*/results.json`
- `src/bispectrum/__init__.py`
- `src/bispectrum/cn_on_cn.py`
- `src/bispectrum/dn_on_dn.py`
- `src/bispectrum/octa_on_octa.py`
- `src/bispectrum/so2_on_disk.py`
- `src/bispectrum/so3_on_s2.py`
- `src/bispectrum/torus_on_torus.py`
- `benchmarks/benchmark.py`
- `pyproject.toml`
- `README.md`
- `.github/workflows/tests.yml`
- `docs/steerable_bispectrum_notes.md`
- `docs/selective_so3.md`
- `docs/inversion_gap_for_octahedral_group.md`
- `docs/per_pixel_steerable_bispectrum.tex`
- `docs/neurips_steerable_bispectrum_plan.md`
- `docs/sanbord_miolane_2025/sections/lit-review.tex`
