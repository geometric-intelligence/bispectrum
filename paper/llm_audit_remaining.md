# LLM-style audit: what is left

The overlay (`\llm` / `\llmpt` marks, `llmaudit.sty`, the index at the end of the PDF) and its
catalogue (`paper/tools/llm_audit.{py,toml}`) have been removed from the paper. This file is the
record of the items that are still open. The annotated state is at commit `a0125bc` on branch
`paper/llm-audit` if the overlay is ever needed again.

181 items were catalogued. 119 are resolved and applied to the text (V 21, R 23, J 26, D 31, S 10, H 8).
62 are left:

| Category | Meaning | Left |
|---|---|---|
| C | Captions and headers | 18 |
| M | Mechanical drift | 34 |
| X | Deferred to the SO(3) completeness pass | 10 |

Line numbers refer to the current `paper/bispectrum.tex` and `paper/checklist.tex`. Section,
figure and table numbers are those of the compiled PDF. Each item quotes the LaTeX it is about
(or, for a point mark, the text just before it) and the options considered.

## C: Captions and headers (18)

### C14a

`bispectrum.tex:178` · §1 Introduction, ¶ Contributions

Point mark, right after:

```tex
…ntributions are: \begin{enumerate}[leftmargin=*] \item \textbf{Software: Pytorch Library.}
```

Bold “Category: Title.” labels on the three contribution items (C14a–c).
- (A) short bold nouns: **Library.**, **Spherical invariant.**, **Evaluation.**
- (B) no bold labels.
- (C) keep.

### C14b

`bispectrum.tex:180` · §1 Introduction, ¶ Contributions

Point mark, right after:

```tex
… coverage). \item \textbf{Theory: Augmented Selective Bispectrum for Spherical Rotations.}
```

Same bold “Category: Title.” pattern (see C14a).

### C14c

`bispectrum.tex:182` · §1 Introduction, ¶ Contributions

Point mark, right after:

```tex
…onjecture that it is generically complete. \item \textbf{Experiments: Unified Evaluation.}
```

Same pattern (see C14a); “Unified Evaluation” is also vague.

### C16

`bispectrum.tex:357` · §3 Related Work, ¶ Invariant theory and scattering

Point mark, right after:

```tex
…ractions and CG power augmentation are built. \paragraph{Invariant theory and scattering.}
```

The header promises invariant theory, but the paragraph covers only scattering.
- (A) **Scattering.**
- (B) add a sentence on invariant theory (polynomial invariants, orbit separation).

### C15

`bispectrum.tex:378` · §3 Related Work, ¶ Invariant theory and scattering, Table 1

Point mark, right after:

```tex
…r}{lcccl} \toprule Nonlinearity & Complete? & Coefficients & Forward cost & Key limitation
```

“Key limitation” column: terse labels (“Scalar bottleneck”, “Proof of concept only”) read as filler, and two of them only restate the cost columns.
- (A) drop the column.
- (B) concrete entries, e.g. “loses relative phase”, “one scalar per irrep”.
- (C) keep.

### C01

`bispectrum.tex:460` · §4 The bispectrum Library, ¶ Profiling, Table 3 caption

Point mark, right after:

```tex
…ond for every module, with up to $512\times$ fewer coefficients than the full bispectrum.}
```

Bold claim used as the caption title; the same pattern runs through most figure and table captions (C01–C10).
- (A) keep, if the style is deliberate.
- (B) a neutral title such as “Forward time and coefficient counts per module”, with the claim moved into the caption text.

### C02

`bispectrum.tex:586` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ Empirical evidence of completeness, Figure 2 caption

Point mark, right after:

```tex
…caption{\textbf{The augmented selective bispectrum keeps the shape of a spherical signal.}
```

Glib caption title (“keeps the shape”).
- (A) “Signals recovered from their 380 invariants match the targets up to rotation.”
- (B) “Reconstruction from the augmented selective bispectrum.”
- (C) keep.

### C11

`bispectrum.tex:597` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ Takeaways

Point mark, right after:

```tex
…~\ref{app:so3-recon}).} \label{fig:so3-reconstruction} \end{figure} \paragraph{Takeaways.}
```

“Takeaways” header (slide-deck style).
- (A) **Results.**
- (B) merge into the previous paragraph.

### C03

`bispectrum.tex:648` · §6.1 $G$-bispectral pooling in $G$-CNNs, ¶ Setup, Figure 3 caption

Point mark, right after:

```tex
…selines on PCam ($G{=}C_8$) and beats max pooling on OrganMNIST3D ($G{=}O$) at full data.}
```

Bold claim caption title (Nina's wording).
- (A) keep.
- (B) neutral title.

### C04

`bispectrum.tex:704` · §6.2 Spherical MNIST: SO(3) invariance on the sphere, ¶ Setup, Table 4 caption

Point mark, right after:

```tex
…xtbf{Spherical MNIST: no accuracy lost to unseen rotations, at a fraction of the latency.}
```

Glib caption title.
- (A) “Spherical MNIST accuracy under three protocols, and latency.”
- (B) keep.

### C05

`bispectrum.tex:720` · §6.2 Spherical MNIST: SO(3) invariance on the sphere, ¶ Setup, Figure 4 caption

Point mark, right after:

```tex
…nted selective bispectrum matches equivariant baselines on Spherical MNIST and is faster.}
```

Bold claim caption title (see C01).
- (A) keep.
- (B) neutral title.

### C12

`bispectrum.tex:765` · §7 Discussion and Conclusion, ¶ When not to use the bispectrum

Point mark, right after:

```tex
… on OrganMNIST3D (Appendix~\ref{app:changes}). \paragraph{When not to use the bispectrum.}
```

Blog-style header.
- (A) merge into Limitations and future work.
- (B) **Scope.**
- (C) keep.

### C13

`bispectrum.tex:920` · §A.2 Why $\mathcal{O}(L)$ is impossible

Point mark, right after:

```tex
…. \end{remark} \subsection{Why $\mathcal{O}(L)$ is impossible} \label{sec:app-lower-bound}
```

Header “Why ... is impossible”: rhetorical question-style header.
- (A) “Lower bound on the output size”.
- (B) “A dimension lower bound”.

### C06

`bispectrum.tex:1327` · §B.1 Coefficient-count scaling, Figure 5 caption

Point mark, right after:

```tex
…cients grows linearly with the group size; for full $G$-bispectra it grows quadratically.}
```

Bold claim caption title (see C01).
- (A) keep.
- (B) neutral title.

### C07

`bispectrum.tex:1342` · §B.2 Forward-pass timing, Figure 6 caption

Point mark, right after:

```tex
…ctive forward pass costs the same at every group size; the full bispectrum grows with it.}
```

Bold claim caption title (see C01).
- (A) keep.
- (B) neutral title.

### C08

`bispectrum.tex:1362` · §B.3 GPU throughput and inversion, Figure 7 caption

Point mark, right after:

```tex
…f samples/s on one H100, and every module except $\mathrm{SO}(3)$ on $S^2$ is invertible.}
```

Bold claim caption title (see C01).
- (A) keep.
- (B) neutral title.

### C09

`bispectrum.tex:1407` · §B.4 Portability across devices and backends, Table 8 caption

Point mark, right after:

```tex
…runs in under a millisecond on every device, except $\SO(3)$ on $S^2$, which needs a GPU.}
```

Bold claim caption title (see C01).
- (A) keep.
- (B) neutral title.

### C10

`bispectrum.tex:1861` · §F.4 Spherical feature ablation, Table 15 caption

Point mark, right after:

```tex
…textbf{Of the augmentation steps, only the Clebsch--Gordan power spectra measurably help.}
```

Bold claim caption title (see C01).
- (A) keep.
- (B) neutral title.

## M: Mechanical drift (34)

### M03a

`bispectrum.tex:143` · Abstract

```tex
band-limit
```

Spelling drift: “band-limit”, “bandlimit” and “band limit” all occur as the noun (eleven marks, M03a–k).
- (A) “band limit” for the noun and “band-limited” for the adjective everywhere.
- (B) “bandlimit” everywhere.

### M11

`bispectrum.tex:144` · Abstract

```tex
from $O(L^3)$ to $\Theta(L^2)$
```

Mixes $O$ and $\Theta$ for the same comparison; the contributions say $\Theta(L^3)$ to $\Theta(L^2)$.
- (A) “from $\Theta(L^3)$ to $\Theta(L^2)$”.

### M01

`bispectrum.tex:178` · §1 Introduction, ¶ Contributions

```tex
Pytorch
```

Typo: “PyTorch”.

### M02

`bispectrum.tex:179` · §1 Introduction, ¶ Contributions

```tex
passes and
```

Double space in the source (“passes and”); invisible in the PDF.
- (A) single space.

### M03b

`bispectrum.tex:181` · §1 Introduction, ¶ Contributions

```tex
bandlimit
```

“bandlimit” (see M03a).

### M06a

`bispectrum.tex:249` · §2 Mathematical Background, ¶ Bispectrum on the disk

```tex
$\mathbb{D}$
```

Disk notation drift: $\mathbb{D}$ here, $D^2$ in the Figure 5 caption.
- (A) $\mathbb{D}$ everywhere.
- (B) $D^2$ everywhere.

### M03c

`bispectrum.tex:259` · §2 Mathematical Background, ¶ Bispectrum on the disk

```tex
bandlimit
```

“bandlimit” (see M03a).

### M03d

`bispectrum.tex:542` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ The augmented selective bispectrum, Algorithm 1

```tex
band-limit
```

“band-limit” (see M03a).

### M08

`bispectrum.tex:620` · §6.1 $G$-bispectral pooling in $G$-CNNs

Point mark, right after:

```tex
…$-bispectral pooling in $G$-CNNs} \label{sec:pooling} \label{sec:pcam} \label{sec:organ3d}
```

Three labels on one subsection (sec:pooling, sec:pcam, sec:organ3d).
- (A) keep `sec:pooling` and point the other references to it.

### M05a

`bispectrum.tex:687` · §6.2 Spherical MNIST: SO(3) invariance on the sphere, ¶ Setup

```tex
$\ell_{\max}=15$
```

$\ell_{\max}$ and $L$ name the same band limit (Section 5 says $L{=}15$).
- (A) $L{=}15$.

### M03e

`bispectrum.tex:773` · §7 Discussion and Conclusion, ¶ Limitations and future work

```tex
band-limits
```

“band-limits” (see M03a).

### M04a

`bispectrum.tex:877` · §A.1 Setup and notation

```tex
realised
```

British spelling; the rest of the paper is American.
- (A) “realized”.

### M07

`bispectrum.tex:919` · §A.2 Why $\mathcal{O}(L)$ is impossible

Point mark, right after:

```tex
…to D^\ell(g)\mathbf{F}_\ell$. \end{remark} \subsection{Why $\mathcal{O}(L)$ is impossible}
```

$\mathcal{O}$ here, $O$ everywhere else.
- (A) $O(L)$.

### M03f

`bispectrum.tex:924` · §A.2 Why $\mathcal{O}(L)$ is impossible

```tex
band-limit
```

“band-limit” (see M03a).

### M03g

`bispectrum.tex:958` · §A.3 The selective index set

```tex
band-limit
```

“band-limit” (see M03a).

### M03h

`bispectrum.tex:1032` · §A.3 The selective index set, Table 6 caption

```tex
band-limits
```

“band-limits” (see M03a).

### M03i

`bispectrum.tex:1063` · §A.3 The selective index set, Algorithm 2 caption

```tex
band-limit
```

“band-limit” (see M03a).

### M03j

`bispectrum.tex:1068` · §A.3 The selective index set, Algorithm 2

```tex
band-limit
```

“band-limit” (see M03a).

### M09

`bispectrum.tex:1193` · §A.4 Parity structure of the bispectrum

```tex
Clebsch-Gordan
```

Single hyphen; elsewhere “Clebsch–Gordan”.
- (A) “Clebsch–Gordan”.

### M06b

`bispectrum.tex:1332` · §B.1 Coefficient-count scaling, Figure 5 caption

```tex
the disk $D^2$
```

Disk notation (see M06a).

### M10a

`bispectrum.tex:1528` · §C.3 Published baselines, Table 9

```tex
Veeling et al.\ 2018
```

Plain-text citation in a table, while other tables use citet (M10a–f).
- (A) use citet.

### M10b

`bispectrum.tex:1529` · §C.3 Published baselines, Table 9

```tex
Veeling et al.\ 2018
```

See M10a.

### M10c

`bispectrum.tex:1530` · §C.3 Published baselines, Table 9

```tex
Veeling et al.\ 2018
```

See M10a.

### M10d

`bispectrum.tex:1531` · §C.3 Published baselines, Table 9

```tex
Veeling et al.\ 2018
```

See M10a.

### M10e

`bispectrum.tex:1552` · §C.3 Published baselines, Table 10

```tex
Yang et al.\ 2023
```

See M10a.

### M10f

`bispectrum.tex:1553` · §C.3 Published baselines, Table 10

```tex
Yang et al.\ 2023
```

See M10a.

### M04b

`bispectrum.tex:1617` · §D.3 Data preprocessing, ¶ PCam

```tex
normalised
```

British spelling (see M04a).
- (A) “normalized”.

### M04c

`bispectrum.tex:1625` · §D.3 Data preprocessing, ¶ OrganMNIST3D

```tex
normalised
```

See M04b.

### M05b

`bispectrum.tex:1636` · §D.3 Data preprocessing, ¶ Spherical MNIST

```tex
$\ell_{\max} = 15$
```

See M05a.
- (A) $L{=}15$.

### M13

`bispectrum.tex:1745` · §E API Example

Point mark, right after:

```tex
…ly available for research use. \end{itemize} \section{API Example} \label{app:api-example}
```

API example: `SO2onDisk` and `OctaonOcta` are imported but unused, and `selective=True` is the default.
- (A) import only `CnonCn` and drop the argument.
- (B) use all three imports.

### M12a

`checklist.tex:8` · Checklist, Claims

```tex
the augmented selective $\SO(3)$ invariant
```

Naming: the paper calls it the augmented selective bispectrum (M12a–c).
- (A) “the augmented selective bispectrum”.

### M12c

`checklist.tex:8` · Checklist, Claims

```tex
$\SO(3)$ invariant
```

Naming (see M12a).

### M12b

`checklist.tex:20` · Checklist, Limitations

```tex
inversion of the spherical invariant
```

Naming (see M12a).
- (A) “inversion of the augmented selective bispectrum”.

### M03k

`checklist.tex:20` · Checklist, Limitations

```tex
band-limits
```

“band-limits” (see M03a).

## X: Deferred to the SO(3) completeness pass (10)

### X05a

`bispectrum.tex:591` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ Empirical evidence of completeness, Figure 2 caption

```tex
the interpolation floor of the alignment
```

Completeness pass: the “interpolation floor” is a median $1.8\times10^{-3}$ here, 0.0014–0.0070 in Takeaways, and 0.004–0.013 in Appendix D; the three do not agree.

### X05b

`bispectrum.tex:601` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ Takeaways

```tex
the interpolation floor of rotating on the grid
```

Completeness pass (see X05a).

### X01

`bispectrum.tex:929` · §A.2 Why $\mathcal{O}(L)$ is impossible

```tex
Zariski-open dense subset
```

Completeness pass: “Zariski-open” applied to a smooth, not necessarily polynomial, map.
- (A) “an open dense subset”.

### X02

`bispectrum.tex:936` · §A.2 Why $\mathcal{O}(L)$ is impossible

```tex
(generic-rank theorem for $\SO(3)$ acting on $V_L$; cf.~Tarski--Seidenberg in the polynomial case)
```

Completeness pass: name-dropping (an unnamed “generic-rank theorem”, Tarski–Seidenberg), and $V_L$ is used before it is defined.
- (A) a direct argument via the constant-rank theorem on the open set where the rank is maximal.
- (B) a specific reference.

### X03a

`bispectrum.tex:942` · §A.2 Why $\mathcal{O}(L)$ is impossible

```tex
pathological non-measurable invariants (e.g.\ a fixed orbit-section indexed by the axiom of choice)
```

Completeness pass: false as stated. Borel-measurable injections from $\mathbb{R}^n$ into $\mathbb{R}$ exist, so the obstruction is continuity, not measurability.
- (A) “discontinuous invariants (e.g. a Borel injection into $\mathbb{R}$)”.
- (B) drop the remark.

### X03b

`bispectrum.tex:945` · §A.2 Why $\mathcal{O}(L)$ is impossible

```tex
All practical bispectral or spectral invariants are polynomial, so the proposition applies.
```

Completeness pass: sweeping claim (“All practical”).
- (A) “The bispectral and CG-power invariants used here are polynomial, so the proposition applies.”
- (B) drop.

### X06

`bispectrum.tex:1277` · §A.5 Comparison with existing results, Table 7

```tex
Yes (constr.)
```

Completeness pass: Section 2 says the operator-bispectrum completeness of Kakarala does not directly apply on $S^2$, so “Yes (constr.)” for the full bispectrum on $S^2$ contradicts the text.

### X04

`bispectrum.tex:1728` · §D.6 Recovering signals from the augmented selective bispectrum (Figure 2), ¶ Results

```tex
recovers all 32 targets exactly
```

Completeness pass: “exactly” conflicts with image errors of 0.0014–0.0070.
- (A) “recovers all 32 targets up to the interpolation error”.

### X05c

`bispectrum.tex:1730` · §D.6 Recovering signals from the augmented selective bispectrum (Figure 2), ¶ Results

```tex
The remaining image error is the interpolation of rotating on the grid:
```

Completeness pass (see X05a).

### X07

`checklist.tex:36` · Checklist, Theory assumptions and proofs

```tex
exact reconstruction at $L{=}15$
```

Completeness pass: “exact” (see X04).

## Not marked: code-speak noticed during the pass

These never got a mark. They are listed so they are not lost.

- `230K params` at `bispectrum.tex:691` (§6.2 Spherical MNIST: SO(3) invariance on the sphere, ¶ Setup). “params” is code-speak, the rest of the paper says “parameters”.
- `emitted by the library` at `bispectrum.tex:1001` (§A.3 The selective index set, Table 5 caption). “emitted” is code-speak (“returned”, or drop).
- `including mandatory even self-couplings` at `bispectrum.tex:1033` (§A.3 The selective index set, Table 6 caption). “mandatory” is implementation jargon.
- `seen by the SMNIST classifier` at `bispectrum.tex:1037` (§A.3 The selective index set, Table 6 caption). “SMNIST” is a code name, the text says “Spherical MNIST”.
- `$\mathrm{proj}(\mathrm{ReLU}(\mathrm{signlog}(\beta)))$` at `bispectrum.tex:1806` (§F.3 Pooling head). `proj` / `signlog` are code names. Define them in words or keep them as notation.
