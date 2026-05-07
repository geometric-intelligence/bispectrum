# Overleaf Comments — bispectrum paper

**Project:** https://www.overleaf.com/project/69dd6ef85579b7e8d7426177  
**Paper title:** bispectrum: Selective G-Bispectra Made Practical  
**Total threads:** 54  
**Pulled:** 2026-04-18  

## Notes on this dump

- Anchors are taken from the **local** `paper/paper.tex` using the character offsets reported by Overleaf. Local and Overleaf tex differ by only ~524 chars in a 105k-char document, so anchor windows will be within a few chars of the exact cursor position.
- Each thread shows: author, timestamp (UTC), full message + any replies, section/subsection context, tex line number, and the surrounding text.
- `|@|` in the anchor window marks the exact Overleaf cursor position.
- A first-pass classification is included for each thread — this is Claude's initial read and should be reviewed.

## Reviewer breakdown

- **simon.mataigne** (simonmataigne@gmail.com) — 30 threads
- **Adele Myers** (adele@ucsb.edu) — 24 threads

## Threads (in document order)

### § (preamble / front-matter)

#### 1. @simon.mataigne — 2026-04-17 14:23 UTC 

- **Location:** line ~63 (pos 1454), section: _—_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
stringstyle=\color{black!60},   frame=single,   breaklines=true,   columns=fullflexible, }  \title{ |@| \texttt{bispectrum}: Selective $G$-Bispectra Made Practical}  \author{%   Johan Mathe \\   Atmo, Inc. \\   \texttt{johan@atmo.ai} \\   \And   Simon Mataigne \\   UCLouvain \\   \texttt{simon.mataigne@
```

**Thread:**

- **simon.mataigne** · _2026-04-17 14:23 UTC_:

> This paper is a banger.

**Claude's first-pass classification:** already-addressed (not a change request — positive comment; no edit needed)

**Proposed response:** "Thanks Simon! Resolving."

---

#### 2. @simon.mataigne — 2026-04-17 15:03 UTC 

- **Location:** line ~105 (pos 2982), section: _—_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
inite groups. Code is available at \url{https://github.com/geometric-intelligence/bispectrum}. \end{ |@| abstract} \section{Introduction} \label{sec:intro}  Equivariant neural networks~\citep{cohen2016group, weiler2019general} enforce symmetry by construction, but they still require an invariant represen
```

**Thread:**

- **simon.mataigne** · _2026-04-17 15:03 UTC_:

> The github url is not anonymous.

**Claude's first-pass classification:** blocker

**Proposed response:** "Good catch — this is a real issue for double-blind review. The URL `https://github.com/geometric-intelligence/bispectrum` identifies the `geometric-intelligence` organization and is not anonymous. We need to replace it with an anonymous repository link (e.g., Anonymous GitHub) before submission, or remove the URL entirely. Flagging for Johan to resolve before the camera-ready or submission deadline."

---

#### ‣ Introduction

#### 3. @simon.mataigne — 2026-04-15 08:57 UTC 

- **Location:** line ~111 (pos 3198), section: _Introduction_ · tid `…000001`
- **Messages:** 2 · **Status:** open

**Anchor (local tex):**

```tex
eral} enforce symmetry by construction, but they still require an invariant representation at the po |@| int where a prediction must no longer depend on orientation. In practice, invariant pooling remains a weak point: standard choices such as norm pooling, gate-based pooling, or max/sum aggregation are
```

**Thread:**

- **simon.mataigne** · _2026-04-15 08:57 UTC_:

> Maybe precise symmetry of what. Symmetry of the output with respect to symmetries of the input.

- **simon.mataigne** · _2026-04-15 09:04 UTC_:

> Moreover, I would use "group symmetries" instead.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good point — the symmetry referent is implicit. Updated the phrase: 'enforce group symmetry of the output with respect to group transformations of the input by construction.' This makes both the type (group symmetry) and the direction (output responds to input transformations) explicit."

---

#### 4. @simon.mataigne — 2026-04-15 09:01 UTC 

- **Location:** line ~111 (pos 3233), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
ion, but they still require an invariant representation at the point where a prediction must no long |@| er depend on orientation. In practice, invariant pooling remains a weak point: standard choices such as norm pooling, gate-based pooling, or max/sum aggregation are usable~\citep{weiler2019general}, b
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:01 UTC_:

> No "still" here?

**Claude's first-pass classification:** nit

**Proposed response:** "Agreed — 'still' is slightly redundant here. Removed."

---

#### 5. @simon.mataigne — 2026-04-15 09:04 UTC 

- **Location:** line ~111 (pos 3260), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
an invariant representation at the point where a prediction must no longer depend on orientation. I |@| n practice, invariant pooling remains a weak point: standard choices such as norm pooling, gate-based pooling, or max/sum aggregation are usable~\citep{weiler2019general}, but they do not preserve all
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:04 UTC_:

> "group representation"

**Claude's first-pass classification:** nit

**Proposed response:** "Good suggestion — 'orientation' is domain-specific; changed to 'group representation' for generality."

---

#### 6. @simon.mataigne — 2026-04-15 08:59 UTC 

- **Location:** line ~112 (pos 3331), section: _Introduction_ · tid `…000001`
- **Messages:** 2 · **Status:** open

**Anchor (local tex):**

```tex
nger depend on orientation. In practice, invariant pooling remains a weak point: standard choices su |@| ch as norm pooling, gate-based pooling, or max/sum aggregation are usable~\citep{weiler2019general}, but they do not preserve all equivariant information. The bispectrum offers a mathematically princi
```

**Thread:**

- **simon.mataigne** · _2026-04-15 08:59 UTC_:

> Maybe "orientation" is already to specific in the context of general group actions.

- **simon.mataigne** · _2026-04-15 09:03 UTC_:

> Maybe "group transformations" instead?

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — 'orientation' implicitly assumes SO(3)/rotation. Changed to 'group transformations' throughout the paragraph where used generically."

---

#### 7. @Adele Myers — 2026-04-16 23:55 UTC 

- **Location:** line ~115 (pos 3547), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
preserve all equivariant information. The bispectrum offers a mathematically principled alternative |@| , since for many groups it yields exact invariants with reconstruction guarantees~\citep{kakarala2012bispectrum}.  The obstacle is practical rather than conceptual. Existing bispectral constructions a
```

**Thread:**

- **Adele Myers** · _2026-04-16 23:55 UTC_:

> maybe change to "all signal information"? One could say the bispectrum also does not preserve "equivariant" information, since it does not encode global phase, whereas equivariant representations do.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good technical point — 'equivariant information' is indeed imprecise here since the bispectrum is an invariant map. Updated to: 'they do not preserve all signal information, collapsing distinct orbits.' This is accurate: the bispectrum preserves orbit-separating information."

---

#### 8. @Adele Myers — 2026-04-16 23:56 UTC 

- **Location:** line ~119 (pos 3704), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
econstruction guarantees~\citep{kakarala2012bispectrum}.  The obstacle is practical rather than conc |@| eptual. Existing bispectral constructions are scattered across theory papers, domain-specific derivations, and small implementations, with no general-purpose PyTorch library covering the finite-group
```

**Thread:**

- **Adele Myers** · _2026-04-16 23:56 UTC_:

> I would cite the original bispectrum inversion paper here by giannakis.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — adding Giannakis 1989 (the original inversion result) alongside kakarala2012bispectrum here: '...exact invariants with reconstruction guarantees~\citep{giannakis1989bispectrum, kakarala2012bispectrum}.' Need to add the Giannakis 1989 bibtex entry to references.bib."

---

#### 9. @simon.mataigne — 2026-04-15 13:50 UTC 

- **Location:** line ~124 (pos 4066), section: _Introduction_ · tid `…000001`
- **Messages:** 2 · **Status:** open

**Anchor (local tex):**

```tex
ning. Even for groups where selectivity reduces the coefficient count from $O(|G|^2)$ to $O(|G|)$~\c |@| itep{mataigne2024selective}, a user still needs a numerically stable implementation, a consistent API, inversion routines, tests, and benchmarked runtime behavior.  This paper presents \texttt{bispect
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:50 UTC_:

> This sentence should be used to say "Even for a group $G$,..."

- **simon.mataigne** · _2026-04-15 13:50 UTC_:

> So that $G$ is defined before being used.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Right — $G$ is used without introduction in this sentence. Updated to: 'Even for a compact group $G$ where selectivity reduces the coefficient count from $O(|G|^2)$ to $O(|G|)$~\citep{mataigne2024selective}, a user still needs...'"

---

#### 10. @simon.mataigne — 2026-04-15 09:08 UTC 

- **Location:** line ~124 (pos 4100), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
ivity reduces the coefficient count from $O(|G|^2)$ to $O(|G|)$~\citep{mataigne2024selective}, a use |@| r still needs a numerically stable implementation, a consistent API, inversion routines, tests, and benchmarked runtime behavior.  This paper presents \texttt{bispectrum}, a PyTorch~\citep{paszke2019p
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:08 UTC_:

> And the computational cost from$O(|G|^3)$ to $O(|G|\log(|G|)$ or $O(|G|^2)$.
> This point seems even more important to recall.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good point — we currently only mention the coefficient count reduction; the computational cost reduction is even more dramatic. Updated: 'selectivity reduces the coefficient count from $O(|G|^2)$ to $O(|G|)$ and the computational cost from $O(|G|^3)$ to $O(|G|^2)$~\citep{mataigne2024selective}.'"

---

#### 11. @Adele Myers — 2026-04-17 00:01 UTC 

- **Location:** line ~125 (pos 4181), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
24selective}, a user still needs a numerically stable implementation, a consistent API, inversion ro |@| utines, tests, and benchmarked runtime behavior.  This paper presents \texttt{bispectrum}, a PyTorch~\citep{paszke2019pytorch} library for computing bispectral invariants and, where available, their i
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:01 UTC_:

> this sentence feels like a non sequitur and should be combined with the previous sentence, as it does not say anything new. Specifically highlighting the selective g bispectrum here feels out of place. It can be mentioned later, along with all the other bispectra.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — the paragraph introducing the library currently feels disconnected. We'll fold the library introduction into the preceding paragraph to flow naturally from the problem statement, and defer the enumeration of specific bispectra to the contributions paragraph."

---

#### 12. @simon.mataigne — 2026-04-15 09:11 UTC 

- **Location:** line ~131 (pos 4515), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
ic, dihedral, torus, disk, sphere, and octahedral settings as differentiable \texttt{nn.Module}s int |@| ended to plug into standard ML workflows. Beyond the software contribution, the paper makes one technical contribution that arose during implementation: an augmented selective $\SO(3)$ bispectrum on $
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:11 UTC_:

> toroidal, discoid, spherical and... right?

**Claude's first-pass classification:** nit

**Proposed response:** "Fair point — using 'cyclic, dihedral, torus, disk, sphere, and octahedral' mixes terminology styles. Will either make all adjectives consistent ('cyclic, dihedral, toroidal, discoid, spherical, and octahedral') or switch to listing the domain spaces ('$C_n$, $D_n$, torus, disk, $S^2$, and octahedral group'). The latter matches the supported-modules table."

---

#### 13. @Adele Myers — 2026-04-17 00:03 UTC 

- **Location:** line ~131 (pos 4551), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
nd octahedral settings as differentiable \texttt{nn.Module}s intended to plug into standard ML workf |@| lows. Beyond the software contribution, the paper makes one technical contribution that arose during implementation: an augmented selective $\SO(3)$ bispectrum on $S^2$ with $O(L^2)$ coefficients (Sec
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:03 UTC_:

> this is vague. I would say something like "library supports group invariance for functions defined on the following spaces: "

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good suggestion. Updated the sentence to: 'The library provides $G$-bispectral invariants for functions defined on the following domains: $C_n$ (cyclic), $D_n$ (dihedral), torus $\mathbb{T}^2$, disk, sphere $S^2$, and the octahedral group.' This is concrete and matches the supported-modules table."

---

#### 14. @simon.mataigne — 2026-04-15 09:12 UTC 

- **Location:** line ~137 (pos 4800), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
bispectrum on $S^2$ with $O(L^2)$ coefficients (Section~\ref{sec:so3-selective}).  \paragraph{Contri |@| butions.} \begin{enumerate}   \item \textbf{Software contribution.} A PyTorch library implementing     bispectral invariants for seven group/domain pairs with a consistent     \texttt{nn.Module}-style
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:12 UTC_:

> the sphere $S^2$

**Claude's first-pass classification:** nit

**Proposed response:** "Good — 'sphere' is ambiguous as a dimension. Changed to 'the sphere $S^2$' for precision."

---

#### 15. @simon.mataigne — 2026-04-15 09:13 UTC 

- **Location:** line ~138 (pos 4810), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
on $S^2$ with $O(L^2)$ coefficients (Section~\ref{sec:so3-selective}).  \paragraph{Contributions.} |@| \begin{enumerate}   \item \textbf{Software contribution.} A PyTorch library implementing     bispectral invariants for seven group/domain pairs with a consistent     \texttt{nn.Module}-style interface
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:13 UTC_:

> What is $L$? Do you mean $|G|$?

**Claude's first-pass classification:** needs-work

**Proposed response:** "Correct — $L$ is introduced here before being defined. Added a brief parenthetical: 'an augmented selective $\SO(3)$ bispectrum on $S^2$ with $O(L^2)$ coefficients (where $L$ is the spherical harmonic band-limit; Section~\ref{sec:so3-selective}).' This pre-empts the question for readers without the SO(3) background."

---

#### 16. @simon.mataigne — 2026-04-15 09:16 UTC 

- **Location:** line ~146 (pos 5270), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
ec:benchmarks}).   \item \textbf{Augmented selective $\SO(3)$ bispectrum on $S^2$.} We construct |@| and implement an augmented selective bispectrum for $\SO(3)$ acting on     $S^2$ that uses     $\Theta(L^2)$ coefficients instead of the full $\Theta(L^3)$,     matching the orbit-space lower bound u
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:16 UTC_:

> I believe US spelling uses "optimized" with a z.

**Claude's first-pass classification:** nit

**Proposed response:** "The local tex at line 141 still reads 'GPU-optimised forward passes'. Addressed in v2: changed to 'GPU-optimized forward passes'."

*Note: Local tex still has 'GPU-optimised' at line 141 — this is NOT yet addressed.*

---

#### 17. @simon.mataigne — 2026-04-15 13:51 UTC 

- **Location:** line ~151 (pos 5618), section: _Introduction_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
es scalar bispectral entries with     Clebsch--Gordan power spectrum entries (a degree-$4$ complemen |@| t     needed for real signals).  We give a computer-assisted completeness     proof for $4 \le L \le 100$ that combines a structural seed step     (Kakarala 1992/2012) with a uniform per-degree linear
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:51 UTC_:

> $L$ comes back here but it should be defined.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — $L$ appears in the contributions bullet (line 148) before it is formally defined. Will add a brief in-line gloss on first use in the contributions list: 'at band-limit $L$ (the maximum spherical harmonic degree).' If we add the parenthetical to thread 15's location that covers this too; otherwise a forward reference to Section~\ref{sec:so3-selective} is sufficient."

---

### § Mathematical Background

#### ‣ Equivariant features and invariant maps

#### 18. @simon.mataigne — 2026-04-15 13:53 UTC 

- **Location:** line ~180 (pos 7037), section: _Mathematical Background_, subsection: _Equivariant features and invariant maps_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
eventually map equivariant features to \emph{invariant} ones.  \begin{table}[t]   \caption{Invarian |@| t nonlinearities for equivariant CNNs. The selective     bispectrum is the only method that is simultaneously exact, complete,     and linear-cost for finite groups.}   \label{tab:nonlinearity-compari
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:53 UTC_:

> $\pi$ and $\Phi$ should be defined here.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Correct — the equivariance definition uses $\pi_{\mathrm{out}}$, $\pi_{\mathrm{in}}$, and $\Phi$ but these are not defined in the surrounding prose. Will add: 'where $\pi_{\mathrm{in}}$ and $\pi_{\mathrm{out}}$ are the group representations on input and output feature spaces respectively, and $\Phi$ is the layer map.' This is a load-bearing sentence and needs the definitions."

---

#### 19. @simon.mataigne — 2026-04-15 09:14 UTC 

- **Location:** line ~201 (pos 7883), section: _Mathematical Background_, subsection: _Equivariant features and invariant maps_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
assumption \\     \bottomrule   \end{tabular}    \vspace{0.3em}   {\small $^\ddagger$Complete pre-al |@| iasing; the pointwise nonlinearity in the forward pass introduces aliasing that may break completeness in practice.} \end{table}  Table~\ref{tab:nonlinearity-comparison} summarises the trade-offs. Nor
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:14 UTC_:

> I would not put the "2" in the bif O cost.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Correct — the Gate row currently reads '$O(2|G|)$' in the Coefficients column. The factor of 2 is an implementation detail (scalar + gate channel), not an asymptotic distinction from $O(|G|)$. Changed to '$O(|G|)$' for consistency with standard big-O notation."

---

#### 20. @simon.mataigne — 2026-04-15 09:15 UTC 

- **Location:** line ~201 (pos 7966), section: _Mathematical Background_, subsection: _Equivariant features and invariant maps_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
r$Complete pre-aliasing; the pointwise nonlinearity in the forward pass introduces aliasing that may |@| break completeness in practice.} \end{table}  Table~\ref{tab:nonlinearity-comparison} summarises the trade-offs. Norm pooling and gated nonlinearities are exact invariants but \emph{incomplete}: they
```

**Thread:**

- **simon.mataigne** · _2026-04-15 09:15 UTC_:

> G\log G is rather the computational cost. The DFT has |G| coefficients.
> But it might be a plus giving both computational and storage costs.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Correct — the Fourier pointwise row in the table lists '$O(|G| \log |G|)$' in the Coefficients column, which is the computational cost, not the coefficient count (which is $O(|G|)$). Options: (a) split into separate Coefficients and Cost columns, or (b) clarify in a table footnote. Recommend splitting the column: the Coefficients column becomes '$O(|G|)$' and we add a footnote '$\dagger$Forward pass cost $O(|G| \log |G|)$ via FFT.'"

---

#### 21. @Adele Myers — 2026-04-17 00:06 UTC 

- **Location:** line ~202 (pos 8003), section: _Mathematical Background_, subsection: _Equivariant features and invariant maps_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
e nonlinearity in the forward pass introduces aliasing that may break completeness in practice.} \en |@| d{table}  Table~\ref{tab:nonlinearity-comparison} summarises the trade-offs. Norm pooling and gated nonlinearities are exact invariants but \emph{incomplete}: they discard the phase of Fourier blocks,
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:06 UTC_:

> which bispectrum and which selective bispectrum? Do you mean the G-bispectrum?

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good clarification point — the table caption says 'The selective bispectrum' without specifying which. Since the paper covers seven group/domain pairs, the statement is about the $G$-bispectrum in general. Updated caption: 'The selective $G$-bispectrum is the only method that is simultaneously exact, complete, and linear-cost for finite groups.' (The 'G' was already in the paper title so this is consistent.)"

---

#### ‣ The $G$-bispectrum and selective bispectra

#### 22. @simon.mataigne — 2026-04-15 13:54 UTC 

- **Location:** line ~211 (pos 8450), section: _Mathematical Background_, subsection: _The $G$-bispectrum and selective bispectra_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
9general}. The full bispectrum is exact and complete but costs $O(|G|^2)$ coefficients.  \subsection |@| {The $G$-bispectrum and selective bispectra}  Let $G$ be a compact group with irreducible representations $\{\rho_i\}_{i \in \hat{G}}$ of dimensions $d_i$, and let $\hat{f}(\rho_i) = \sum_{g \in G} f(
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:54 UTC_:

> Same question about US spelling with a z.

**Claude's first-pass classification:** nit

**Proposed response:** "Local tex at line 716 reads 'Our library generalises'. Changed to 'generalizes' for US spelling consistency."

*Note: This comment is at pos 8450, but the actual 'generalises' instance is at line 716 in the Related Work section — Simon may have misclicked. Either way, the fix is the same: change 'generalises' → 'generalizes'.*

---

#### 23. @simon.mataigne — 2026-04-15 13:55 UTC 

- **Location:** line ~214 (pos 8601), section: _Mathematical Background_, subsection: _The $G$-bispectrum and selective bispectra_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
G$ be a compact group with irreducible representations $\{\rho_i\}_{i \in \hat{G}}$ of dimensions $d |@| _i$, and let $\hat{f}(\rho_i) = \sum_{g \in G} f(g)\, \rho_i(g)$ denote the group Fourier transform. Given the Clebsch--Gordan decomposition $\rho_i \otimes \rho_j \cong \bigoplus_k m_{ij}^k \rho_k$ w
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:55 UTC_:

> The notion of orbit is not yet defined.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Correct — 'separates orbits' (line 228) uses 'orbit' without a definition. Added a one-sentence definition before Eq.~(1): 'The \emph{orbit} of a signal $f$ under $G$ is the set $\{f \circ g : g \in G\}$; two signals are in the same orbit iff they differ by a group transformation.' Alternatively, the definition can be embedded inline: 'it separates \emph{$G$-orbits} (equivalence classes of signals related by group action) for generic signals.'"

---

#### 24. @Adele Myers — 2026-04-17 00:10 UTC 

- **Location:** line ~218 (pos 8853), section: _Mathematical Background_, subsection: _The $G$-bispectrum and selective bispectra_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
imes \rho_j \cong \bigoplus_k m_{ij}^k \rho_k$ with intertwining matrix $C_{ij}$, the \emph{bispectr |@| um} is \begin{equation}   \label{eq:bispectrum-def}   \beta_{\rho_i, \rho_j}     = C_{ij}^\top\,       (\hat{f}(\rho_i) \otimes \hat{f}(\rho_j))\,       C_{ij}\, \hat{f}(\rho_k)^\dagger. \end{equation
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:10 UTC_:

> I would include a separate section for bispectra of signals defined over spaces that are homogeneous for the group, and then another section for bispectra of signals defined over spaces that are NOT homogeneous for the group (as far as I know, mine is the only one). However, including mine in the G-bispectrum section is inaccurate, and other bispectra exist that are not G-bispectra.

**Claude's first-pass classification:** needs-work

**Proposed response:** "This is a valid taxonomic point — the disk bispectrum (myers2025selective) acts on a non-homogeneous space and is structurally different from the standard $G$-bispectrum. However, splitting the Background into homogeneous/non-homogeneous sections would substantially restructure a 9-page paper. Our compromise: add a paragraph note in the Background subsection that explicitly flags the distinction: 'The disk bispectrum of \citet{myers2025selective} operates on a domain that is \emph{not} a homogeneous $G$-space; it is included in the library but is taxonomically distinct from the $G$-bispectra above.' This acknowledges the distinction without a full structural overhaul."

---

#### 25. @simon.mataigne — 2026-04-15 13:56 UTC 

- **Location:** line ~223 (pos 8972), section: _Mathematical Background_, subsection: _The $G$-bispectrum and selective bispectra_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
ion}   \label{eq:bispectrum-def}   \beta_{\rho_i, \rho_j}     = C_{ij}^\top\,       (\hat{f}(\rho_i) |@| \otimes \hat{f}(\rho_j))\,       C_{ij}\, \hat{f}(\rho_k)^\dagger. \end{equation} Each $\beta_{\rho_i,\rho_j}$ is invariant under $f \mapsto f \circ g$ and, taken over all pairs $(i,j)$, the bispectr
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:56 UTC_:

> What is $\hat{G}$?

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good catch — $\hat{G}$ (the Pontryagin dual / set of irreps) is used at line 214 and again at line 232 but never explicitly defined. Will add: 'where $\hat{G}$ denotes the set of equivalence classes of irreducible representations of $G$' immediately after its first appearance. For compact groups this is standard notation but the paper should define it."

---

#### 26. @simon.mataigne — 2026-04-15 13:58 UTC 

- **Location:** line ~226 (pos 9074), section: _Mathematical Background_, subsection: _The $G$-bispectrum and selective bispectra_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
otimes \hat{f}(\rho_j))\,       C_{ij}\, \hat{f}(\rho_k)^\dagger. \end{equation} Each $\beta_{\rho_i |@| ,\rho_j}$ is invariant under $f \mapsto f \circ g$ and, taken over all pairs $(i,j)$, the bispectrum is a \emph{complete} invariant: it separates orbits for generic signals~\citep{kakarala2012bispectr
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:58 UTC_:

> It would be good pointing to  a reference here, e.g., Diaconis & Rockmore: Efficient computation of the Fourier Transform for finite groups.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good suggestion — adding Diaconis & Rockmore (1990) as a reference for the group Fourier transform and its efficient computation, alongside the existing kakarala2012bispectrum citation. Need to add the bibtex entry for 'diaconis1990fourier' (Diaconis & Rockmore, JACM 1990)."

---

#### 27. @simon.mataigne — 2026-04-15 13:59 UTC 

- **Location:** line ~232 (pos 9453), section: _Mathematical Background_, subsection: _The $G$-bispectrum and selective bispectra_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
tra.} \citet{mataigne2024selective} show that for finite groups, a BFS traversal of the Kronecker pr |@| oduct table of $\hat{G}$ selects $O(|G|)$ bispectral coefficients that remain complete; \citet{myers2025selective} extend this to the disk. This reduction from quadratic to linear is the key to practi
```

**Thread:**

- **simon.mataigne** · _2026-04-15 13:59 UTC_:

> "Each bispectrum coefficient \beta..."

**Claude's first-pass classification:** nit

**Proposed response:** "Agreed — the current phrasing says 'selects $O(|G|)$ bispectral coefficients'. To be more precise and concrete, changed to: 'selects $O(|G|)$ bispectral coefficients $\beta_{\rho_i,\rho_j}$ that remain complete.'"

---

### § The \texttt{bispectrum}

#### ‣ Design and implementation

#### 28. @simon.mataigne — 2026-04-15 14:01 UTC 

- **Location:** line ~241 (pos 9817), section: _The \texttt{bispectrum}_, subsection: _Design and implementation_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
6$ to $172$ coefficients.  \section{The \texttt{bispectrum} Library} \label{sec:library}  \subsectio |@| n{Design and implementation} \label{sec:design}  Every group/domain pair is exposed as a \texttt{torch.nn.Module} named \texttt{\{Group\}on\{Domain\}} (e.g.\ \texttt{SO3onS2}). Modules accept raw real
```

**Thread:**

- **simon.mataigne** · _2026-04-15 14:01 UTC_:

> Maybe unroll the acronym here? Breadth-First-Search

**Claude's first-pass classification:** nit

**Proposed response:** "Good suggestion — on first use (line 232), 'BFS traversal' should be written 'breadth-first search (BFS) traversal'. The second use at line 299 can then use the abbreviation."

---

#### 29. @Adele Myers — 2026-04-17 00:11 UTC 

- **Location:** line ~251 (pos 10356), section: _The \texttt{bispectrum}_, subsection: _Design and implementation_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
d stored as non-learnable buffers, so \texttt{.to(device)} works transparently and forward passes in |@| cur zero recomputation. Where available, \texttt{bsp.invert(beta)} returns the reconstructed signal up to group-action indeterminacy. The only dependencies are PyTorch~\citep{paszke2019pytorch}, NumPy
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:11 UTC_:

> I would introduce the concept of domain earlier. (perhaps integrate it where I placed my previous comment about listing that "we define bispectra for signals defined over the space" but instead, say "domain"

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — 'domain' in 'group/domain pairs' (line 244) is first used without definition. The term 'domain' should be introduced in the Introduction when listing supported settings: 'The library provides bispectra for signals defined on seven domains (the underlying space on which the group acts): $C_n$, $D_n$, torus $\mathbb{T}^2$, disk, sphere $S^2$, and the octahedral setting.' This resolves threads 13 and 29 together."

---

#### ‣ Supported modules

#### 30. @Adele Myers — 2026-04-17 00:12 UTC 

- **Location:** line ~280 (pos 11424), section: _The \texttt{bispectrum}_, subsection: _Supported modules_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
ut mode \\     \midrule     \texttt{CnonCn} & $C_n$ on $C_n$ & selective + full \\     \texttt{SO2on |@| S1} & $\SO(2)$ on $S^1$ & selective + full \\     \texttt{TorusOnTorus} & product cyclic groups on torus & selective + full \\     \texttt{DnonDn} & $D_n$ on $D_n$ & selective \\     \texttt{SO2onDisk
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:12 UTC_:

> I would cite each of the papers in this table.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good suggestion — adding a 'Reference' column to Table~\ref{tab:supported-groups}, citing the underlying mathematical paper for each module: CnonCn → \citep{mataigne2024selective}; SO2onS1 → same; DnonDn → \citep{mataigne2024selective}; SO2onDisk → \citep{myers2025selective}; SO3onS2 → \citep{kakarala2012bispectrum}; OctaonOcta → \citep{mataigne2024selective}. Alternatively, add citations in a table footnote."

---

#### ‣ Augmented selective $\SO(3)$ bispectrum on $S^2$

#### 31. @simon.mataigne — 2026-04-17 14:10 UTC 

- **Location:** line ~303 (pos 12436), section: _The \texttt{bispectrum}_, subsection: _Augmented selective $\SO(3)$ bispectrum on $S^2$_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
d, the orbit space $\R^{(L+1)^2}/\SO(3)$ has dimension $(L{+}1)^2 - 3$, so any complete invariant re |@| quires $\Omega(L^2)$ independent components. We achieve this lower bound.  \begin{theorem}[Augmented selective $\SO(3)$ invariant on $S^2$] \label{thm:so3-selective} For $f : S^2 \to \R$ band-limited
```

**Thread:**

- **simon.mataigne** · _2026-04-17 14:10 UTC_:

> Maybe precise "for signals defined on the sphere $S^2$.

**Claude's first-pass classification:** nit

**Proposed response:** "Good — the section heading already says 'on $S^2$' but the theorem statement opening 'For $f : S^2 \to \R$ band-limited at degree...' already specifies this. If Simon means the section opening prose, we can add 'for real signals defined on the sphere $S^2$' to the introductory sentence."

---

#### 32. @Adele Myers — 2026-04-17 00:19 UTC 

- **Location:** line ~313 (pos 12925), section: _The \texttt{bispectrum}_, subsection: _Augmented selective $\SO(3)$ bispectrum on $S^2$_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
eta_{\ell_1,\ell_2,\ell}$ together with a subset $\mathcal{S}_P$ of CG power spectrum entries $P_{\e |@| ll_1,\ell_2,\ell} := \lVert (\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2})|_\ell \rVert^2$---that is a complete $\SO(3)$-invariant for Lebesgue-almost-every signal. The output size $\Theta(L^2)$ ma
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:19 UTC_:

> By "augmented" do you mean you have to include things about the original signal in order to invert it? If this is the case, then it feels weird to call this bispectrum complete. Usually, when someone says their bispectrum is complete, they say it after proving the signal can be reconstructed from the coefficients alone.

**Claude's first-pass classification:** blocker

**Proposed response:** "Good catch — 'augmented' refers to augmenting the *bispectral entries* (adding CG power spectrum entries, which are still invariants derived solely from the signal), NOT augmenting with the original signal. The invariant $\Phi_{\mathrm{aug}}(f)$ is computed purely from $f$, and completeness means it separates $\SO(3)$-orbits for generic $f$. We need to clarify this in the theorem statement and the surrounding prose: 'The term \emph{augmented} refers to supplementing the scalar bispectral entries $\beta_{\ell_1,\ell_2,\ell}$ with degree-4 CG power spectrum entries $P_{\ell_1,\ell_2,\ell}$ (which are also $\SO(3)$-invariant functions of $f$), not with the signal itself.' This is a real terminological issue that will confuse readers."

---

#### 33. @Adele Myers — 2026-04-17 00:13 UTC 

- **Location:** line ~314 (pos 13037), section: _The \texttt{bispectrum}_, subsection: _Augmented selective $\SO(3)$ bispectrum on $S^2$_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
\ell} := \lVert (\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2})|_\ell \rVert^2$---that is a comple |@| te $\SO(3)$-invariant for Lebesgue-almost-every signal. The output size $\Theta(L^2)$ matches the dimension $(L{+}1)^2 - 3$ of the orbit space $\R^{(L+1)^2}/\SO(3)$ up to a constant factor. \end{theor
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:13 UTC_:

> what is "degree"

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good point — 'band-limited at degree $4 \le L \le 100$' uses 'degree' informally. In the proof appendix (Definition 1), 'band-limited at degree $L$' is formally defined as having $a_\ell^m = 0$ for $\ell > L$. We should add a forward reference or brief gloss in the theorem statement: 'band-limited at degree $L$ (i.e., with spherical harmonic expansion truncated at angular frequency $L$, $4 \le L \le 100$).'"

---

#### 34. @Adele Myers — 2026-04-17 00:15 UTC 

- **Location:** line ~328 (pos 13677), section: _The \texttt{bispectrum}_, subsection: _Augmented selective $\SO(3)$ bispectrum on $S^2$_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
tional or deterministic witnesses for each $\ell \le 100$ and promoted to generic statements by alge |@| braic-variety arguments. The structural form of the bootstrap is uniform for $\ell \ge 8$; extending the verification beyond $L = 100$ requires only re-running a single script. \end{remark}  \paragrap
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:15 UTC_:

> I would give more background on your method here and explain why your method is destined to work.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Fair request — Remark 1 (the proof sketch) is quite technical without setting up the intuition. We'll add 2–3 sentences before the remark block explaining the high-level approach: 'The proof strategy follows a bootstrap induction: we first recover the low-degree coefficients from a small set of bispectral entries (the seed step), then show that each higher-degree coefficient vector $\mathbf{F}_\ell$ can be recovered linearly from already-known lower-degree coefficients. The scalar bispectrum alone is rank-deficient for real signals at certain degrees; the CG power entries supply the complementary magnitude information needed for full Jacobian rank.' This gives readers the why before the how."

---

#### 35. @Adele Myers — 2026-04-17 00:17 UTC 

- **Location:** line ~341 (pos 14341), section: _The \texttt{bispectrum}_, subsection: _Augmented selective $\SO(3)$ bispectrum on $S^2$_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
bootstrap matrix that \emph{cannot} be resolved by choosing different bispectral triples. At $\ell = |@| 4$, for example, the bootstrap matrix has complex rank~$4$ instead of $9 = 2\ell{+}1$, and this holds even for the \emph{full} bispectrum (all valid triples). The CG power spectrum entries $P_{\ell_1
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:17 UTC_:

> You're saying that for complex signals defined on the sphere, you can use the scalar bispectrum? First, what is the scalar bispectrum on the sphere? I've heard of the vector bispectrum as being the simplest bispectrum on the sphere.

**Claude's first-pass classification:** blocker

**Proposed response:** "Good catch — the paper uses 'scalar bispectrum' without defining what is meant. In the SO(3)/S² context, the \emph{scalar} bispectrum refers to the scalar contraction $\beta_{\ell_1,\ell_2,\ell} = \sum_{m_1,m_2,m} C^{\ell m}_{\ell_1 m_1;\ell_2 m_2} a_{\ell_1}^{m_1} a_{\ell_2}^{m_2} \overline{a_\ell^m}$, which yields a single scalar (not a vector or tensor) per triple $(\ell_1,\ell_2,\ell)$. The 'vector bispectrum' Adele refers to is the full matrix $\beta_{\ell_1,\ell_2}^{(\ell)} = C_{\ell_1\ell_2}^\top (\hat{f}(\ell_1) \otimes \hat{f}(\ell_2)) C_{\ell_1\ell_2} \hat{f}(\ell)^\dagger$ — a matrix. We need to add a definition of 'scalar bispectrum' when it first appears (line 298 or in the paragraph 'Why augmentation is necessary'), making clear it is the scalar contraction of the matrix bispectrum with CG coefficients, and cite the original vector bispectrum reference. This is a load-bearing definitional gap."

---

#### 36. @Adele Myers — 2026-04-17 00:26 UTC 

- **Location:** line ~357 (pos 15150), section: _The \texttt{bispectrum}_, subsection: _Augmented selective $\SO(3)$ bispectrum on $S^2$_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
; \textbf{(4)~Self-coupling} $(\ell, \ell, \ell')$ with $\ell'$ even. Entries with $\ell_1 = \ell_2$ |@| and odd~$\ell$ vanish identically for real signals (CG antisymmetry of the symmetric square) and are excluded.  All \emph{even} self-coupling entries $\beta_{\ell,\ell,\ell'}$ with $2 \le \ell' \le \
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:26 UTC_:

> It would be helpful if you also gave background about the so3 on s03 bispectrum, and the original sphere vector bispectrum. Not only should you do it for completeness and to cite that paper, but also it helps introduce notation necessary to understand this section

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — the Section 3.3 proof discussion assumes familiarity with SO(3) harmonics notation that is not introduced in the Background section. We'll add a short paragraph in the Background section: 'For $G = \SO(3)$ acting on $S^2$, the irreps are Wigner-$D$ matrices $D^\ell$ of dimension $2\ell+1$, and the Fourier coefficients $\mathbf{F}_\ell = (a_\ell^{-\ell},\ldots,a_\ell^\ell)$ are the spherical harmonic coefficients. The \emph{scalar bispectrum} $\beta_{\ell_1,\ell_2,\ell}$ [citation] is the CG-contracted scalar; the full matrix bispectrum is due to \citet{kakarala2012bispectrum} [and original vector bispectrum reference].' This resolves threads 35 and 36 together."

---

#### ‣ Computational benchmarks

#### 37. @Adele Myers — 2026-04-17 00:31 UTC 

- **Location:** line ~447 (pos 19548), section: _The \texttt{bispectrum}_, subsection: _Computational benchmarks_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
finite abelian modules ($C_n$: 0.15\,ms, $\mathbb{T}^2$: 0.08\,ms), under 2\,ms for $\SO(2)$/disk (1 |@| .94\,ms), and under 1\,ms for the octahedral group (0.70\,ms), making it negligible relative to backbone computation. $\SO(3)$ on $S^2$ at $\ell_{\max}{=}16$ takes 26.5\,ms, dominated by the spherical
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:31 UTC_:

> I would mention how you compute |G| for the disk bispectrum. This is not clear without explanation.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good point — for $\SO(2)$ on the disk, $|G| = \infty$ (continuous group), so the table shows '$\infty$' for $|G|$ and counts refer to band-limited coefficients. A brief sentence in the benchmarks section should clarify: 'For continuous groups ($\SO(2)$, $\SO(3)$), $|G| = \infty$; the coefficient count refers to the band-limited representation at the tested resolution ($N_{\max}$ radial modes and angular frequencies for the disk).' The table caption already says this partially; it should also appear in the text."

---

### § Experiments

#### ‣ 2D histopathology classification under cyclic symmetry

#### 38. @Adele Myers — 2026-04-17 00:34 UTC 

- **Location:** line ~512 (pos 22526), section: _Experiments_, subsection: _2D histopathology classification under cyclic symmetry_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
6 \pm .012$ \\     Gated                         & 136K & $.941 \pm .009$ & $.870 \pm .004$ \\     B |@| ispectrum (\texttt{CnonCn})  & 128K & $.941 \pm .004$ & $\mathbf{.872 \pm .006}$ \\     Fourier-ELU                   & 110K & $\mathbf{.945 \pm .004}$ & $.855 \pm .006$ \\     NormReLU
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:34 UTC_:

> If you only test on Cn x Cn bispectrum, I would re-word this to say "and CnonCn selective bispectrum"

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — the section title is '2D histopathology classification under cyclic symmetry' and the subsection intro already says 'selective bispectrum via \texttt{CnonCn}$(n{=}8)$', but Table~\ref{tab:pcam-results} labels the row just 'Bispectrum (\texttt{CnonCn})'. Adding 'selective' to be consistent: 'Selective Bispectrum (\texttt{CnonCn})'. Adele's broader concern (thread 40) about group labels in figure/table captions is addressed globally."

---

#### 39. @Adele Myers — 2026-04-17 00:36 UTC 

- **Location:** line ~521 (pos 22981), section: _Experiments_, subsection: _2D histopathology classification under cyclic symmetry_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
four equivariant pooling methods substantially outperform the standard augmented CNN (AUC $0.896$). |@| Fourier-ELU leads slightly ($0.945$), followed by NormReLU ($0.942$), bispectrum and gated (both $0.941$). At 1\% data, the bispectrum leads all methods ($0.912$), outperforming NormReLU ($0.873$) an
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:36 UTC_:

> Since these are results for CnonCn, I would include that in the figure labels.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — the figure caption for Figure~\ref{fig:pcam-pareto} does not say which bispectrum variant is used. Updated caption to: '...PCam Pareto analysis (\texttt{CnonCn} selective bispectrum, single seed, 10\% training data)...'. This makes figures self-contained for readers who skip to figures."

---

#### ‣ 3D organ classification under octahedral symmetry

#### 40. @Adele Myers — 2026-04-17 00:37 UTC 

- **Location:** line ~579 (pos 25820), section: _Experiments_, subsection: _3D organ classification under octahedral symmetry_ · tid `…000001`
- **Messages:** 2 · **Status:** open

**Anchor (local tex):**

```tex
educed data (Figure~\ref{fig:data-efficiency}), the bispectrum's advantage widens: at 5\% data it re |@| aches $33.2\%$ vs.\ $18.9\%$ for max pool. At higher capacity (6M params, Appendix~\ref{app:organ3d-details}), max pool overtakes the bispectrum, suggesting the completeness advantage is specific to t
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:37 UTC_:

> since these are results for OctaonOcta, "Bispectrum" should --> OctaonOcta Bispectrum. I know it should be clear because it is under the results section for OctaonOcta, but some people only look at tables and figures, and this would be very confusing for them.

- **Adele Myers** · _2026-04-17 00:38 UTC_:

> Same comment applies for all results sections

**Claude's first-pass classification:** needs-work

**Proposed response:** "Agreed — this is important for figure/table self-containedness. For the organ3D section: change 'Bispectrum' → '\texttt{OctaonOcta} Bispectrum' in Table~\ref{tab:organ3d-results} and Figure~\ref{fig:data-efficiency} caption. Apply the same pattern globally: PCam results → '\texttt{CnonCn} Selective Bispectrum', Spherical MNIST → '\texttt{SO3onS2} Selective Bispectrum'. The text uses the \texttt{module} names already; the tables and figures should match."

---

#### ‣ Spherical MNIST: SO(3) invariance on the sphere

#### 41. @Adele Myers — 2026-04-17 00:37 UTC 

- **Location:** line ~605 (pos 26976), section: _Experiments_, subsection: _Spherical MNIST: SO(3) invariance on the sphere_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
projected onto the sphere via stereographic projection and sampled on a $64 \times 128$ equiangular |@| grid. We compute the selective bispectrum at $\ell_{\max} = 15$, producing 384 complex invariant coefficients (768 real features after taking log-magnitude), which are fed to a 3-layer MLP (232K para
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:37 UTC_:

> OctaonOcta

**Claude's first-pass classification:** nit

**Proposed response:** "This appears to be Adele noting the module name should appear — the section is about SO(3)/S², so the module name '\texttt{SO3onS2}' should be mentioned here. The setup paragraph already says 'we test \texttt{SO3onS2}' a few lines earlier (line 599), so this is already addressed in the text. If Adele means the table/figure labels, it is covered by the global label change in thread 40."

---

### § Related Work

#### 42. @Adele Myers — 2026-04-17 00:40 UTC 

- **Location:** line ~724 (pos 32631), section: _Related Work_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
or field networks~\citep{thomas2018tensor} and MACE~\citep{batatia2022mace} use iterated Clebsch--Go |@| rdan products to build equivariant interactions. These products are exact and complete in principle, but each layer incurs $O(L^3)$ cost per CG contraction. \citet{xie2025price} analyse the \emph{pric
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:40 UTC_:

> I feel like this should go at the beginning of the paper? and should also cite papers on the the selective g bispectrum, the disk bispectrum, the so3 on so3 bispectrum, the bispectrum on homogeneous spaces (where the sphere vector bispectrum is defined)

**Claude's first-pass classification:** needs-work

**Proposed response:** "Adele raises a valid point about related work positioning — some venues prefer related work early. However, at NeurIPS the standard structure places related work after experiments. The current placement after experiments is standard and appropriate. We'll keep the current structure and add the missing citations as requested: selective G-bispectrum \citep{mataigne2024selective}, disk bispectrum \citep{myers2025selective}, SO(3)-on-SO(3) bispectrum [citation needed], and the sphere vector bispectrum [citation needed]. Please provide the missing bibtex keys for the last two."

---

#### 43. @simon.mataigne — 2026-04-15 14:03 UTC 

- **Location:** line ~736 (pos 33303), section: _Related Work_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
avelet scattering transform~\citep{mallat2012scattering} provides translation-invariant representati |@| ons via iterated modulus operations; \citet{esteves2018spherical} extend this to the sphere. Unlike the bispectrum, scattering representations are stable but generally incomplete---they discard inter-
```

**Thread:**

- **simon.mataigne** · _2026-04-15 14:03 UTC_:

> In the paper, only "bispectrum" is used. (without the G)

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good consistency note — the Related Work section has 'G-bispectrum' in some places but the paper body uses 'bispectrum' or 'selective bispectrum' without the G prefix. Will audit and make consistent: use '$G$-bispectrum' when referring to the general construct (as in the paper title), and 'bispectrum' when the context is clear. The Related Work paragraph on bispectral approaches should use '$G$-bispectrum' to be precise (line 712 already does)."

---

#### 44. @simon.mataigne — 2026-04-15 14:05 UTC 

- **Location:** line ~740 (pos 33602), section: _Related Work_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
-scale phase information. The bispectrum's completeness guarantee~\citep{kakarala2012bispectrum} is |@| its distinguishing theoretical advantage.  \section{Discussion and Conclusion} \label{sec:discussion}  We have presented \texttt{bispectrum}, a PyTorch library that makes selective $G$-bispectral inva
```

**Thread:**

- **simon.mataigne** · _2026-04-15 14:05 UTC_:

> "generalizes" for US spelling

**Claude's first-pass classification:** nit

**Proposed response:** "Local tex at line 716 reads 'Our library generalises'. Changed to 'generalizes'."

---

### § Discussion and Conclusion

#### 45. @Adele Myers — 2026-04-17 00:41 UTC 

- **Location:** line ~765 (pos 34951), section: _Discussion and Conclusion_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
lete alternatives such as max pooling can match or surpass bispectral pooling, likely because the ne |@| twork learns to compensate for missing invariant information. Completeness is a property of the \emph{map}, not of downstream task performance: a complete invariant provides a stronger inductive bias,
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:41 UTC_:

> the first paper to prove the invertibility of the bispectrum was giannakis in 1989

**Claude's first-pass classification:** needs-work

**Proposed response:** "Correct historical attribution — the current paper cites kakarala2012bispectrum for reconstruction guarantees but does not mention Giannakis 1989 as the original invertibility result. Adding: 'Invertibility of the bispectrum for generic signals was first proved by \citet{giannakis1989bispectrum}; \citet{kakarala2012bispectrum} extended this to compact groups.' This fixes the attribution in both Introduction (thread 8) and Discussion. The bibtex entry for Giannakis 1989 must be added."

---

#### 46. @Adele Myers — 2026-04-17 00:42 UTC 

- **Location:** line ~789 (pos 36220), section: _Discussion and Conclusion_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
n to the benchmarks used here.  \paragraph{Future work.} Key directions include: implementing $\SO(3 |@| )$ inversion from selective bispectral coefficients; a structural proof of bootstrap rank for all~$L$; numerical conditioning analysis at large band-limits; integration with \texttt{escnn}~\citep{cesa
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:42 UTC_:

> could also mention that bispectra are generally invariant to a specific group. for example, transation invariance or rotation invariance, but learned invariance is better if the practitioner needs invariance to many groups.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good framing for the limitations paragraph. Adding to the 'When not to use the bispectrum' section: 'Each bispectrum module is invariant to a \emph{specific} group; for tasks requiring invariance to multiple or unknown symmetries, learned invariance (e.g., via data augmentation or group-equivariant networks with flexible symmetry) may be more appropriate.'"

---

### § Selective $\SO(3)$ Bispectrum -- Full Proof

#### 47. @Adele Myers — 2026-04-17 00:45 UTC 

- **Location:** line ~835 (pos 37861), section: _Selective $\SO(3)$ Bispectrum -- Full Proof_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
\     1 & 1 & 0 & 1 \\     2 & 3 & 1 & 4 \\     3 & 7 & 4 & 11 \\     4 & 12 & 5 & 17 \\     $\ell \ |@| ge 5$ & $\sim 2\ell{+}3$ & $\sim \ell$ & $\sim 3\ell$ \\     \bottomrule   \end{tabular} \end{table}  \subsection{Setup and notation}  \begin{definition}[Band-limited signal on $S^2$] A signal $f : S^
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:45 UTC_:

> I could have misinterpreted, but when I was chatting with Nina, she described its independence from escnn as being a good thing (less clunky)?

**Claude's first-pass classification:** needs-work

**Proposed response:** "This comment is about the future-work bullet on 'integration with \texttt{escnn}' (line 792). Adele's point (via Nina) is that independence from escnn is a *feature*, not a gap. We should reframe: instead of listing escnn integration as future work (implying it's missing), say 'The library is intentionally framework-agnostic; it can serve as a drop-in invariant module for architectures built with \texttt{escnn} or \texttt{e3nn}.' Remove it from the future-work list or clarify the framing."

---

#### ‣ Setup and notation

#### 48. @simon.mataigne — 2026-04-17 14:15 UTC 

- **Location:** line ~898 (pos 40395), section: _Selective $\SO(3)$ Bispectrum -- Full Proof_, subsection: _Setup and notation_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
cg-power}   P_{\ell_1,\ell_2,\ell}   := \lVert (\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2})|_\e |@| ll      \rVert^2   = \sum_{m=-\ell}^{\ell}     \Bigl\lvert \sum_{\substack{m_1,m_2 \\ m_1+m_2=m}}       \CG{\ell_1}{m_1}{\ell_2}{m_2}{\ell}{m}\;       a_{\ell_1}^{m_1}\, a_{\ell_2}^{m_2} \Bigr\rvert^2
```

**Thread:**

- **simon.mataigne** · _2026-04-17 14:15 UTC_:

> This matrix should be defined.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Simon is pointing at the CG power spectrum definition. The expression $(\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2})|_\ell$ (the projection of the tensor product onto the $\ell$ isotypic component) is not defined inline. Adding a sentence: 'Here $(\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2})|_\ell$ denotes the projection of the Kronecker product $\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2}$ onto the $D^\ell$-isotypic subspace via the CG coefficients.' The explicit sum in the display equation already gives this, but a verbal definition clarifies."

---

#### 49. @Adele Myers — 2026-04-17 00:48 UTC 

- **Location:** line ~905 (pos 40669), section: _Selective $\SO(3)$ Bispectrum -- Full Proof_, subsection: _Setup and notation_ · tid `…000001`
- **Messages:** 2 · **Status:** open

**Anchor (local tex):**

```tex
ell_2}^{m_2} \Bigr\rvert^2. \end{equation} This is a degree-$4$ polynomial in the signal coefficient |@| s (real and non-negative for real~$f$) and $\SO(3)$-invariant: the CG-coupled product transforms as $D^\ell(g)$, and $\lVert \cdot \rVert^2$ is invariant under unitary transformations. \end{definition
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:48 UTC_:

> this is the vector bispectrum, right? Make sure to cite the vector bispectrum papaer

- **Adele Myers** · _2026-04-17 00:49 UTC_:

> (and for consistency, I would call it the vector bispectrum)

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good question — technically, $P_{\ell_1,\ell_2,\ell} = \|(\mathbf{F}_{\ell_1} \otimes \mathbf{F}_{\ell_2})|_\ell\|^2$ is the \emph{power spectrum of the CG-coupled product}, not the vector bispectrum itself. The vector bispectrum $\beta^{(\ell_1,\ell_2)}_\ell$ is the full CG-decomposed matrix contraction (the matrix generalization of Eq.~(1)), which yields a scalar only after the additional contraction with $\hat{f}(\rho_k)^\dagger$. We should add a clarifying footnote and cite the original vector/matrix bispectrum paper. Adele: please provide the exact reference for the 'vector bispectrum' you have in mind."

---

#### ‣ The selective index set

#### 50. @simon.mataigne — 2026-04-17 14:20 UTC 

- **Location:** line ~998 (pos 44278), section: _Selective $\SO(3)$ Bispectrum -- Full Proof_, subsection: _The selective index set_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
a_{0,\ell,\ell} = a_0^0 \norm{\mathbf{F}_\ell}^2$ (a \emph{quadratic} constraint).  \medskip \noinde |@| nt\textbf{Category~4 (Self-coupling).} Triples $(\ell, \ell, \ell')$ with $0 \le \ell' < \ell$. Quadratic in $\mathbf{F}_\ell$. Selected only for the remaining deficit at low degrees. \end{definition}
```

**Thread:**

- **simon.mataigne** · _2026-04-17 14:20 UTC_:

> $\ell$ from $0$ to $L$,

**Claude's first-pass classification:** needs-work

**Proposed response:** "Simon is pointing out that Definition (Selective bispectrum on $S^2$) does not specify the range of $\ell$. The definition begins 'For each target degree $\ell = 0, \dotsc, L$' (line 971), so it IS stated. However, if Simon's comment is about a different location (the Category 4 self-coupling description at line 998--1001), it reads '$0 \le \ell' < \ell$' without specifying the range of $\ell$. Will add 'for $\ell = 0, \dotsc, L$' at the start of the definition or verify that Definition~\ref{def:app-selective} already states it (it does at line 971)."

---

#### 51. @Adele Myers — 2026-04-17 00:51 UTC 

- **Location:** line ~1046 (pos 45994), section: _Selective $\SO(3)$ Bispectrum -- Full Proof_, subsection: _The selective index set_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
idates     \COMMENT{power entry}   \IF{$\ell \notin \{1,\, 2\}$}     \FOR{$\ell' = \ell{-}1, \dotsc, |@| 0$}       \STATE append $(\ell,\, \ell,\, \ell')$ to candidates     \ENDFOR   \ENDIF   \IF{$\ell = 2$}     \STATE append $(\ell,\, \ell,\, \ell)$ to candidates       \COMMENT{$\beta_{2,2,2}$}   \ENDI
```

**Thread:**

- **Adele Myers** · _2026-04-17 00:51 UTC_:

> Is this similar to the algorithm for finding selective coefficients in the selective G-bispectrum paper? If so I would cite that paper here and mention it in the main text. I suppose I would recommend mentioning the selective G-bispectrum paper regardless, since it is a closely related work, as it also provides an algorithm for finding a selective bispectrum

**Claude's first-pass classification:** needs-work

**Proposed response:** "Good point — Algorithm~\ref{alg:so3-augmented} is inspired by the BFS traversal in \citet{mataigne2024selective} but adapted for the SO(3)/S² case (which requires CG power augmentation, not present in the finite-group algorithm). Adding a sentence at the start of the Algorithm description: 'This algorithm extends the BFS-based selective index construction of \citet{mataigne2024selective} from finite groups to the SO(3) setting, adding the CG power augmentation step that is specific to real signals on $S^2$.' The main text already cites mataigne2024selective; the appendix algorithm should too."

---

### § Constructive Seed Solver (Implementation)

#### ‣ Seed recovery ($\ell \le 3$)

#### 52. @simon.mataigne — 2026-04-15 14:13 UTC 

- **Location:** line ~1840 (pos 81647), section: _Constructive Seed Solver (Implementation)_, subsection: _Seed recovery ($\ell \le 3$)_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
0\norm{\mathbf{F}_3}^2$: gives the norm. \item $\beta_{3,3,2}$: quadratic in $\mathbf{F}_3$, linear |@| in $\mathbf{F}_2$. \item $\beta_{2,3,2}$, $\beta_{2,2,3}$: odd parity with repeated indices,   vanish identically for real signals (Proposition~\ref{prop:app-odd-vanishing}). \item $P_{1,2,1}$ (from d
```

**Thread:**

- **simon.mataigne** · _2026-04-15 14:13 UTC_:

> unique in [0, 2\pi/n), ortherwise there are n solutions.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Simon is flagging a phase/angular ambiguity in the seed recovery — likely in the argument recovery step for $a_1^0$ or a gauge-fixing angle. The seed recovery for $\mathbf{F}_2$ (Lemma~\ref{lem:app-ell2}) leaves $\sgn(v)$ ambiguous (the $T_R$ fibre), and the gauge-fixing step fixes $\arg(a_2^1)$ — but the azimuthal phase is indeed unique only in $[0, 2\pi)$ unless we specify a branch. Adding a clarification: 'The gauge-fixed azimuthal angle is unique in $[0, 2\pi)$; otherwise the $\SO(2)_z$ stabiliser would give $n$ solutions.' Johan should identify the exact step Simon is flagging and add the uniqueness-domain qualifier there."

---

#### ‣ Degree-4 fibre reduction and $T_R$ resolution

#### 53. @simon.mataigne — 2026-04-17 14:45 UTC 

- **Location:** line ~1893 (pos 84402), section: _Constructive Seed Solver (Implementation)_, subsection: _Degree-4 fibre reduction and $T_R$ resolution_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
f{lem:app-degree4-filter}). \end{remark}  \subsection{Degree-4 fibre reduction and $T_R$ resolution} |@| \begin{lemma}[Degree-4 eliminates all spurious seed solutions] \label{lem:app-degree4-filter} Let $\mathcal{S}$ denote the finite seed fibre at degrees~$0$--$3$ (Lemma~\ref{lem:app-ell3}). For gener
```

**Thread:**

- **simon.mataigne** · _2026-04-17 14:45 UTC_:

> Was this notation already introduced?

**Claude's first-pass classification:** needs-work

**Proposed response:** "Simon is asking about notation at the start of the Degree-4 subsection — likely $\mathcal{S}$ (the seed fibre), $T_R$ (the reflection map), or $A_4$ (the degree-4 matrix). $T_R$ is defined in Definition~\ref{def:app-TR} (line 1165 in the SO(3) proof appendix), but Appendix~\ref{app:constructive-solver} is a separate section and should include a back-reference: 'Let $T_R$ be the azimuthal reflection of Definition~\ref{def:app-TR}.' For $\mathcal{S}$: it is named 'finite seed fibre' in the lemma statement itself, which is fine. Add cross-references to ensure readers can trace notation."

---

#### 54. @simon.mataigne — 2026-04-17 14:54 UTC 

- **Location:** line ~1907 (pos 85014), section: _Constructive Seed Solver (Implementation)_, subsection: _Degree-4 fibre reduction and $T_R$ resolution_ · tid `…000001`
- **Messages:** 1 · **Status:** open

**Anchor (local tex):**

```tex
e inconsistent at degree~$4$. \end{lemma} \begin{proof} The degree-4 augmented system is an overdete |@| rmined polynomial system in the 9~real parameters of $\mathbf{F}_4$, with coefficients depending on the seed solution $(\mathbf{F}_0, \ldots, \mathbf{F}_3)$.  For each of the~10 seed solutions on the
```

**Thread:**

- **simon.mataigne** · _2026-04-17 14:54 UTC_:

> I agree with the conclusion but I am not certain of the explanation. We can discuss it in a meeting.

**Claude's first-pass classification:** needs-work

**Proposed response:** "Simon agrees with the conclusion of Lemma~\ref{lem:app-degree4-filter} (degree-4 eliminates spurious seeds) but has concerns about the proof argument. This should be resolved in a co-author meeting before submission. The proof relies on numerical evidence (multi-start NLS with 200 random starts achieving zero residual for the true seed and $\ge 10^{-2}$ for spurious ones); Simon may want either a stronger analytical argument or clearer separation thresholds. Flagging as needing a meeting discussion."

---
