# LLM-style audit: what is left

The overlay (`\llm` / `\llmpt` marks, `llmaudit.sty`, the index at the end of the PDF) and its
catalogue (`paper/tools/llm_audit.{py,toml}`) have been removed from the paper. This file is the
record of the items that are still open. The annotated state is at commit `a0125bc` on branch
`paper/llm-audit` if the overlay is ever needed again.

181 items were catalogued. 171 are resolved and applied to the text (V 21, R 23, J 26, D 31, S 10,
H 8, C 18, M 34), along with the unmarked code-speak noticed during the pass. The 10 left are the
SO(3) completeness pass:

| Category | Meaning | Left |
|---|---|---|
| X | Deferred to the SO(3) completeness pass | 10 |

Line numbers refer to the current `paper/bispectrum.tex` and `paper/checklist.tex`. Section,
figure and table numbers are those of the compiled PDF. Each item quotes the LaTeX it is about
and the options considered.

## X: Deferred to the SO(3) completeness pass (10)

### X05a

`bispectrum.tex:594` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ Empirical evidence of completeness, Figure 2 caption

```tex
the interpolation floor of the alignment
```

Completeness pass: the “interpolation floor” is a median $1.8\times10^{-3}$ here, 0.0014–0.0070 in Takeaways, and 0.004–0.013 in Appendix D; the three do not agree.

### X05b

`bispectrum.tex:604` · §5 The Augmented Selective Bispectrum on $S^2$, ¶ Results

```tex
the interpolation floor of rotating on the grid
```

Completeness pass (see X05a).

### X01

`bispectrum.tex:930` · §A.2 Lower bound on the output size

```tex
Zariski-open dense subset
```

Completeness pass: “Zariski-open” applied to a smooth, not necessarily polynomial, map.
- (A) “an open dense subset”.

### X02

`bispectrum.tex:937` · §A.2 Lower bound on the output size

```tex
(generic-rank theorem for $\SO(3)$ acting on $V_L$; cf.~Tarski--Seidenberg in the polynomial case)
```

Completeness pass: name-dropping (an unnamed “generic-rank theorem”, Tarski–Seidenberg), and $V_L$ is used before it is defined.
- (A) a direct argument via the constant-rank theorem on the open set where the rank is maximal.
- (B) a specific reference.

### X03a

`bispectrum.tex:943` · §A.2 Lower bound on the output size

```tex
pathological non-measurable invariants (e.g.\ a fixed orbit-section indexed by the axiom of choice)
```

Completeness pass: false as stated. Borel-measurable injections from $\mathbb{R}^n$ into $\mathbb{R}$ exist, so the obstruction is continuity, not measurability.
- (A) “discontinuous invariants (e.g. a Borel injection into $\mathbb{R}$)”.
- (B) drop the remark.

### X03b

`bispectrum.tex:946` · §A.2 Lower bound on the output size

```tex
All practical bispectral or spectral invariants are polynomial, so the proposition applies.
```

Completeness pass: sweeping claim (“All practical”).
- (A) “The bispectral and CG-power invariants used here are polynomial, so the proposition applies.”
- (B) drop.

### X06

`bispectrum.tex:1279` · §A.5 Comparison with existing results, Table 7

```tex
Yes (constr.)
```

Completeness pass: Section 2 says the operator-bispectrum completeness of Kakarala does not directly apply on $S^2$, so “Yes (constr.)” for the full bispectrum on $S^2$ contradicts the text.

### X04

`bispectrum.tex:1730` · §D.6 Recovering signals from the augmented selective bispectrum (Figure 2), ¶ Results

```tex
recovers all 32 targets exactly
```

Completeness pass: “exactly” conflicts with image errors of 0.0014–0.0070.
- (A) “recovers all 32 targets up to the interpolation error”.

### X05c

`bispectrum.tex:1732` · §D.6 Recovering signals from the augmented selective bispectrum (Figure 2), ¶ Results

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
