# Spherical MNIST Classification

MNIST digits projected onto S2, classified using SO(3)-invariant bispectral features vs. power spectrum and standard CNN baselines.

```bash
pip install -e "../../[dev]"
python train.py --model bispectrum --train_mode NR
./run_sweep.sh  # full sweep: 3 models x 2 modes x 3 seeds
```
