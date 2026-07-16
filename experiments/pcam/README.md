# PatchCamelyon Classification

Binary classification on PatchCamelyon (96x96 histopathology patches) comparing bispectral pooling against norm, gate, FourierELU, and norm-pool (power-spectrum-like) baselines in equivariant DenseNets.

```bash
pip install -e "../../[dev]"
python train.py --model bispectrum --group c8 --data_dir ./pcam_data
./run_matched_sweep.sh      # Pareto sweep: 6 models x 5 growth rates x 3 seeds
./run_data_pareto_sweep.sh  # AUC-vs-train-size curves at matched ~100K params
```
