# Spherical MNIST Reconstruction

Reconstruct spherical MNIST digits from their augmented selective SO(3) invariant via gradient descent, providing empirical evidence about orbit-information preservation. This is not a global completeness proof.

```bash
pip install -e "../../[dev,experiments]"
python reconstruct.py --n_digits 8 --n_rotations 2
```
