"""Verify the hardcoded octahedral group tables satisfy group axioms.

These tables are correctness-critical for OctaonOcta but are large literals
transcribed from a paper / generator script — there is no other test that
would catch a single-character typo in `_CAYLEY`, `_INVERSE`, or
`_RHO4_SIGNS`. The tests here recompute everything from `_ELEMENTS_3x3`
(themselves verifiable as 24 distinct orthogonal matrices with det +1) and
compare.

Reference: Mataigne et al., NeurIPS 2024, Appendix E.
"""

import torch

from bispectrum.octa_on_octa import _CAYLEY, _INVERSE, _RHO4_SIGNS, _ELEMENTS_3x3

_IDENTITY_INDEX = 23  # g23 == identity by construction


class TestElementsAreValidGroup:
    def test_24_elements(self) -> None:
        assert _ELEMENTS_3x3.shape == (24, 3, 3)

    def test_all_elements_are_rotations(self) -> None:
        for i, g in enumerate(_ELEMENTS_3x3):
            assert torch.allclose(g @ g.T, torch.eye(3, dtype=g.dtype), atol=1e-12), f'g_{i} is not orthogonal'
            assert torch.isclose(torch.det(g), torch.tensor(1.0, dtype=g.dtype), atol=1e-12), (
                f'g_{i} has det != +1 (improper rotation, not in SO(3))'
            )

    def test_elements_are_distinct(self) -> None:
        flat = _ELEMENTS_3x3.reshape(24, 9)
        for i in range(24):
            for j in range(i + 1, 24):
                assert not torch.allclose(flat[i], flat[j], atol=1e-12), (
                    f'g_{i} and g_{j} are equal — table has a duplicate'
                )

    def test_identity_is_g23(self) -> None:
        assert torch.allclose(_ELEMENTS_3x3[_IDENTITY_INDEX], torch.eye(3, dtype=torch.float64))


def _matrix_index(g: torch.Tensor, atol: float = 1e-10) -> int:
    """Find the index k such that _ELEMENTS_3x3[k] == g."""
    for k in range(24):
        if torch.allclose(_ELEMENTS_3x3[k], g, atol=atol):
            return k
    raise AssertionError(f'Matrix not in _ELEMENTS_3x3:\n{g}')


class TestCayleyTable:
    def test_shape_and_dtype(self) -> None:
        assert _CAYLEY.shape == (24, 24)
        assert _CAYLEY.dtype == torch.long

    def test_matches_matrix_multiplication(self) -> None:
        """_CAYLEY[i, j] must equal index_of(g_i @ g_j) for all i, j."""
        for i in range(24):
            for j in range(24):
                product = _ELEMENTS_3x3[i] @ _ELEMENTS_3x3[j]
                expected = _matrix_index(product)
                assert _CAYLEY[i, j].item() == expected, (
                    f'_CAYLEY[{i}, {j}] = {_CAYLEY[i, j].item()} but g_{i} @ g_{j} = g_{expected}'
                )

    def test_associative(self) -> None:
        """(g_i · g_j) · g_k == g_i · (g_j · g_k) for all triples."""
        for i in range(24):
            for j in range(24):
                for k in range(24):
                    left = _CAYLEY[_CAYLEY[i, j], k]
                    right = _CAYLEY[i, _CAYLEY[j, k]]
                    assert left == right, (
                        f'Associativity broken at i={i}, j={j}, k={k}: left={left.item()}, right={right.item()}'
                    )

    def test_identity_acts_as_identity(self) -> None:
        e = _IDENTITY_INDEX
        for i in range(24):
            assert _CAYLEY[e, i].item() == i, f'_CAYLEY[e, g_{i}] != g_{i}'
            assert _CAYLEY[i, e].item() == i, f'_CAYLEY[g_{i}, e] != g_{i}'

    def test_each_row_is_a_permutation(self) -> None:
        """In a Cayley table every row and column is a permutation of {0..23}."""
        expected = set(range(24))
        for i in range(24):
            assert set(_CAYLEY[i, :].tolist()) == expected, f'Row {i} is not a permutation'
            assert set(_CAYLEY[:, i].tolist()) == expected, f'Col {i} is not a permutation'


class TestInverseTable:
    def test_shape_and_dtype(self) -> None:
        assert _INVERSE.shape == (24,)
        assert _INVERSE.dtype == torch.long

    def test_left_and_right_inverse(self) -> None:
        e = _IDENTITY_INDEX
        for i in range(24):
            inv = _INVERSE[i].item()
            assert _CAYLEY[i, inv].item() == e, f'g_{i} @ g_{i}^{{-1}} != e'
            assert _CAYLEY[inv, i].item() == e, f'g_{i}^{{-1}} @ g_{i} != e'

    def test_inverse_is_involution(self) -> None:
        """(g^{-1})^{-1} == g."""
        for i in range(24):
            assert _INVERSE[_INVERSE[i]].item() == i

    def test_inverse_matches_transpose(self) -> None:
        """For rotation matrices R^{-1} = R^T."""
        for i in range(24):
            inv = _INVERSE[i].item()
            assert torch.allclose(_ELEMENTS_3x3[inv], _ELEMENTS_3x3[i].T, atol=1e-12)


class TestRho4Signs:
    def test_shape_and_values(self) -> None:
        assert _RHO4_SIGNS.shape == (24,)
        for s in _RHO4_SIGNS.tolist():
            assert s in (1.0, -1.0), f'_RHO4_SIGNS contains a non-±1 value: {s}'

    def test_identity_has_sign_plus_one(self) -> None:
        assert _RHO4_SIGNS[_IDENTITY_INDEX].item() == 1.0

    def test_homomorphism(self) -> None:
        """rho4 must be a 1-D representation: rho4(g_i g_j) == rho4(g_i) * rho4(g_j)."""
        for i in range(24):
            for j in range(24):
                expected = _RHO4_SIGNS[i].item() * _RHO4_SIGNS[j].item()
                actual = _RHO4_SIGNS[_CAYLEY[i, j]].item()
                assert actual == expected, (
                    f'rho4 is not a homomorphism at i={i}, j={j}: '
                    f'rho4(g_i g_j) = {actual} but rho4(g_i) * rho4(g_j) = {expected}'
                )
