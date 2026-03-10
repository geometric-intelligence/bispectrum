# The Inversion Gap for the Octahedral Group

**Target audience**: Authors of *Selective G-Bispectrum* (Mataigne et al., NeurIPS 2024)

## Summary

The sequential bootstrap algorithm (Algorithms 1–3 in the paper) correctly inverts the selective bispectrum for $C_n$, commutative groups, and $D_n$. When applied to the octahedral group $O$ (Section 8.1 of the paper), the same procedure **fails at recovering $\\mathcal{F}\_{\\rho_2}$ and $\\mathcal{F}\_{\\rho_3}$** from $\\beta\_{\\rho_1,\\rho_1}$. The bispectrum itself is fine — the 172-coefficient selective $O$-Bispectrum is complete. What breaks is the *algorithm* for reconstructing the Fourier coefficients.

This document explains precisely why.

## The bootstrap strategy (recap)

All three algorithms share the same skeleton:

1. Recover $\\mathcal{F}\_{\\rho_0}$ from $\\beta\_{\\rho_0,\\rho_0} = \\mathcal{F}\_{\\rho_0}^3$.
2. Recover $\\mathcal{F}\_{\\rho_1}$ from $\\frac{\\beta\_{\\rho_0,\\rho_1}}{\\mathcal{F}\_{\\rho_0}} = \\mathcal{F}\_{\\rho_1}^\\dagger \\mathcal{F}\_{\\rho_1}$, up to a unitary/orthogonal ambiguity $Q$.
3. Extract higher Fourier coefficients from subsequent bispectral coefficients using CG matrices to block-diagonalize tensor products.

Step 2 always introduces an ambiguity: $\\mathcal{F}\_{\\rho_1} = S \\cdot Q$ where $S = (\\mathcal{F}\_{\\rho_1}^\\dagger \\mathcal{F}\_{\\rho_1})^{1/2}$ is determined, but $Q$ is a free orthogonal matrix. For the algorithm to work, this ambiguity must not corrupt step 3.

## Why it works for $C_n$ and $D_n$

### $C_n$: Phase ambiguity in $\\mathrm{SO}(2)$

The irreps are 1D: $\\rho_k(g) = e^{2\\pi i k g / n}$. The ambiguity in $\\mathcal{F}\_{\\rho_1}$ is a phase $e^{i\\varphi}$, $\\varphi \\in \[0, 2\\pi)$. Step 3 computes:

$$\\mathcal{F}\_{\\rho\_{k+1}} = \\left(\\frac{\\beta\_{\\rho_1, \\rho_k}}{\\mathcal{F}\_{\\rho_1} \\mathcal{F}\_{\\rho_k}}\\right)^\\dagger$$

The phase $e^{i\\varphi}$ propagates coherently through all coefficients as $e^{i\\varphi k}$, which corresponds to a continuous shift $\\Theta(g) \\to \\Theta(g + \\varphi)$. The paper correctly identifies this (Section 5, Appendix A) and resolves it by finding the unique $\\varphi$ that makes the reconstructed signal real. No information is lost.

### $D_n$: $\\mathrm{O}(2)$ ambiguity absorbed by CG matrices

The generating irrep $\\rho_1$ is 2D. Recovery gives $\\mathcal{F}\_{\\rho_1} = V\\Lambda^{1/2}V^\\dagger U$ with $U \\in \\mathrm{O}(2)$. Step 3 extracts higher coefficients via:

$$\\bigoplus\_{\\rho \\in \\rho_1 \\otimes \\rho\_{k-1}} \\mathcal{F}\_\\rho = \\left(C\_{\\rho_1,\\rho\_{k-1}}^\\dagger \\left[\\mathcal{F}\_{\\rho_1} \\otimes \\mathcal{F}\_{\\rho\_{k-1}}\\right]^{-1} \\beta\_{\\rho_1,\\rho\_{k-1}} C\_{\\rho_1,\\rho\_{k-1}}\\right)^\\dagger$$

This extraction relies on the CG matrix $C$ block-diagonalizing the tensor product. The implicit question is: does the unknown $U$ corrupt the extraction?

**It does not**, because the 2D irreps of $D_n$ are restrictions of $\\mathrm{SO}(2)$ irreps. The CG matrices for $D_n$ therefore block-diagonalize $\\rho(U)$ for *all* $U \\in \\mathrm{O}(2)$, not just $U \\in D_n$. Concretely, $C^\\dagger(U \\otimes U)C$ remains block-diagonal for any $U \\in \\mathrm{O}(2)$. The $\\mathrm{O}(2)$ ambiguity passes cleanly through the CG extraction.

## Why it fails for $O$

### The $\\mathrm{SO}(3)$ ambiguity is not absorbed

For the octahedral group, $\\rho_1$ is the standard 3D representation. From $\\beta\_{\\rho_0, \\rho_1}$ we recover:

$$\\mathcal{F}\_{\\rho_1}^\\top \\mathcal{F}\_{\\rho_1} = \\frac{\\beta\_{\\rho_0,\\rho_1}}{\\mathcal{F}\_{\\rho_0}}$$

Taking the symmetric square root gives $S = (\\mathcal{F}\_{\\rho_1}^\\top \\mathcal{F}\_{\\rho_1})^{1/2}$, with the true coefficient being $\\mathcal{F}\_{\\rho_1} = Q \\cdot S$ for some unknown $Q \\in \\mathrm{O}(3)$.

The next step extracts $\\mathcal{F}\_{\\rho_2}$ and $\\mathcal{F}\_{\\rho_3}$ from $\\beta\_{\\rho_1, \\rho_1}$ using the CG matrix $C\_{11}$ that block-diagonalizes $\\rho_1 \\otimes \\rho_1$:

$$C\_{11}^\\top \\cdot \\beta\_{\\rho_1,\\rho_1} \\cdot (S \\otimes S)^{-1} \\cdot C\_{11} = C\_{11}^\\top (Q \\otimes Q) C\_{11} \\cdot \\bigoplus_k \\mathcal{F}\_{\\rho_k}^\\top$$

The CG matrix $C\_{11}$ is constructed to block-diagonalize $\\rho_1(g) \\otimes \\rho_1(g)$ for $g \\in O$. That is:

$$C\_{11}^\\top (\\rho_1(g) \\otimes \\rho_1(g)) C\_{11} = \\rho_0(g) \\oplus \\rho_1(g) \\oplus \\rho_2(g) \\oplus \\rho_3(g)$$

for all $g \\in O$. But $Q$ is **not** constrained to be an element of $O$. It is an arbitrary element of $\\mathrm{SO}(3)$, which is a continuous group with 3 parameters, while $O$ has only 24 elements. For a generic $Q \\in \\mathrm{SO}(3) \\setminus O$, the matrix $C\_{11}^\\top (Q \\otimes Q) C\_{11}$ is **not block-diagonal**. It has cross-block leakage between the $\\rho_2$ and $\\rho_3$ sectors.

### Cross-block leakage: the concrete failure

The Kronecker decomposition of $\\rho_1 \\otimes \\rho_1$ for $O$ is:

$$\\rho_1 \\otimes \\rho_1 = \\rho_0 \\oplus \\rho_1 \\oplus \\rho_2 \\oplus \\rho_3$$

with dimensions $1 + 3 + 3 + 2 = 9$. The CG matrix $C\_{11}$ is $9 \\times 9$ orthogonal. In this basis, the diagonal blocks are:

| Block | Rows | Irrep                   | Dim |
| ----- | ---- | ----------------------- | --- |
| 1     | 0    | $\\rho_0$ (trivial)     | 1   |
| 2     | 1–3  | $\\rho_1$ (standard 3D) | 3   |
| 3     | 4–6  | $\\rho_2$ (product 3D)  | 3   |
| 4     | 7–8  | $\\rho_3$ (2D)          | 2   |

When we compute $C\_{11}^\\top (Q \\otimes Q) C\_{11}$ for a **group element** $Q = \\rho_1(g)$, $g \\in O$, the off-diagonal blocks (e.g., rows 4–6 × columns 7–8) are zero. This is exactly what the CG matrix guarantees.

When $Q$ is a generic rotation, e.g.,

$$Q = R_x(\\theta) = \\begin{pmatrix} 1 & 0 & 0 \\ 0 & \\cos\\theta & -\\sin\\theta \\ 0 & \\sin\\theta & \\cos\\theta \\end{pmatrix}, \\quad \\theta = 0.7 \\text{ (not a multiple of } \\pi/2\\text{)}$$

then $C\_{11}^\\top (Q \\otimes Q) C\_{11}$ has nonzero entries in the $\\rho_2$–$\\rho_3$ cross-block. Numerically, for $\\theta = 0.7$:

$$|[C\_{11}^\\top (Q \\otimes Q) C\_{11}]\_{4{:}7, ; 7{:}9}|\_\\infty \\approx 0.788$$

This is not a small perturbation — it is $O(1)$. When we try to extract $\\mathcal{F}\_{\\rho_2}$ from the rows 4–6 diagonal block, we get a mixture of $\\rho_2$ and $\\rho_3$ contributions. The recovered Fourier coefficients are wrong.

### Why this doesn't happen for $D_n$

The 2D irreps of $D_n$ are of the form $\\rho_k(a^j) = \\begin{pmatrix} \\cos(2\\pi jk/n) & -\\sin(2\\pi jk/n) \\ \\sin(2\\pi jk/n) & \\cos(2\\pi jk/n) \\end{pmatrix}$. These are restrictions of $\\mathrm{SO}(2)$ representations to $\\mathbb{Z}/n\\mathbb{Z}$.

The CG decomposition of $\\rho_j \\otimes \\rho_k$ for $D_n$ has a special property: the CG matrices also block-diagonalize $R\_\\theta \\otimes R\_\\theta$ for **any** $R\_\\theta \\in \\mathrm{SO}(2)$. This is because the tensor product of two $\\mathrm{SO}(2)$ irreps decomposes as $\\rho_j \\otimes \\rho_k = \\rho\_{j+k} \\oplus \\rho\_{|j-k|}$, and this decomposition holds for the full continuous group, not just the finite subgroup. The CG matrices are inherited from $\\mathrm{SO}(2)$ and respect arbitrary rotations.

For $O \\subset \\mathrm{SO}(3)$, the situation is fundamentally different. The irreps of $O$ are **not** restrictions of $\\mathrm{SO}(3)$ irreps (except $\\rho_1$). The CG matrices are specific to the 24-element group and do not block-diagonalize arbitrary $\\mathrm{SO}(3)$ rotations.

## Connection to the paper's own remarks

The paper acknowledges a related issue in the commutative group setting (Appendix C, around Eq. 10):

> *"Again, the indeterminacy factor $h_l$ is not restricted to $\\mathbb{Z}/n_l\\mathbb{Z}$ but can belong to $[0, n_l]$. We will have to solve this issue further."*

This is the same phenomenon in a different guise. For commutative groups, the continuous indeterminacy is scalar (a phase) and can be resolved by enforcing real-valuedness. For $O$, the continuous indeterminacy is a $3 \\times 3$ rotation matrix $Q \\in \\mathrm{SO}(3)$, and there is no analogous closed-form trick to pin it down.

The octahedral section (Section 8.1) says:

> *"We apply the procedure from Algorithms 1, 2 and 3 to Table 6."*

and states that $\\mathcal{F}\_{\\rho_1}$ is recovered "up to an indeterminacy which is a transformation in $\\mathrm{O}(3)$." But the subsequent extraction of $\\mathcal{F}\_{\\rho_2}, \\mathcal{F}\_{\\rho_3}$ implicitly assumes this $\\mathrm{O}(3)$ ambiguity is absorbed by the CG matrices — which, as shown above, it is not.

## What works instead

The selective $O$-Bispectrum **is** complete (Theorem 5 of the paper). The 172 coefficients uniquely determine the signal up to $O$-action. The issue is purely algorithmic: the sequential bootstrap doesn't produce the right answer because of cross-block contamination.

We have implemented a working inversion using:

1. **Bootstrap initialization**: recover $\\mathcal{F}\_{\\rho_0}$ exactly, $\\mathcal{F}\_{\\rho_1}$ up to $Q$, and approximate $\\mathcal{F}\_{\\rho_2}, \\mathcal{F}\_{\\rho_3}, \\mathcal{F}\_{\\rho_4}$ from diagonal blocks (ignoring leakage). This gives a decent starting point.
2. **Levenberg-Marquardt correction**: use forward-mode autodiff through the bispectrum computation to solve $|\\beta(f\_\\text{rec}) - \\beta\_\\text{target}|^2 = 0$ via damped least-squares steps. Each step is a single linear solve, not iterative optimization.
3. **Multi-start**: randomize the initial $Q$ across several restarts to escape local minima.

This converges reliably and is fully differentiable (compatible with `torch.func.jacfwd`).

## Numerical disproof (code)

The following code demonstrates the cross-block leakage. It constructs the CG matrix $C\_{11}$ for $\\rho_1 \\otimes \\rho_1$, applies a generic $Q \\in \\mathrm{SO}(3)$, and shows the off-diagonal blocks are nonzero.

```python
import torch
from bispectrum.o_on_r3 import OonR3

bsp = OonR3()
C11 = bsp._cg_11.to(torch.float64)

# Generic SO(3) rotation (not in O)
theta = 0.7
Q = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
    [0.0, torch.sin(torch.tensor(theta)),  torch.cos(torch.tensor(theta))],
], dtype=torch.float64)

QQ = torch.kron(Q, Q)
M = C11.T @ QQ @ C11

# rho2 block: rows 4:7, rho3 block: rows 7:9
cross_block = M[4:7, 7:9]
print(f"Cross-block (rho2 x rho3) max entry: {cross_block.abs().max():.6f}")
# Output: ~0.788 — far from zero

# Compare with a group element
rho1_g4 = bsp._get_irrep_mats(1)[4].to(torch.float64)
QQ_group = torch.kron(rho1_g4, rho1_g4)
M_group = C11.T @ QQ_group @ C11
cross_block_group = M_group[4:7, 7:9]
print(f"Cross-block for g∈O: {cross_block_group.abs().max():.6f}")
# Output: ~0.0 (machine precision)
```

## Summary table

| Group | Generating irrep dim | Ambiguity in $\\mathcal{F}\_{\\rho_1}$ | CG absorbs it?                  | Bootstrap works? |
| ----- | -------------------- | -------------------------------------- | ------------------------------- | ---------------- |
| $C_n$ | 1                    | $e^{i\\varphi} \\in \\mathrm{SO}(2)$   | N/A (scalar)                    | Yes              |
| $D_n$ | 2                    | $U \\in \\mathrm{O}(2)$                | Yes (CG from $\\mathrm{SO}(2)$) | Yes              |
| $O$   | 3                    | $Q \\in \\mathrm{O}(3)$                | **No**                          | **No**           |
