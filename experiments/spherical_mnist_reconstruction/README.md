# Spherical MNIST Reconstruction

Reconstruct spherical MNIST digits from their SO(3) bispectrum via gradient descent, demonstrating completeness of the bispectral invariant up to SO(3) orbit indeterminacy.

```bash
pip install -e "../../[dev]"
python reconstruct.py --n_digits 8 --n_rotations 2
```
