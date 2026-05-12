# PatchCamelyon Classification

Binary classification on PatchCamelyon (96x96 histopathology patches) comparing bispectral pooling against norm, gate, and FourierELU nonlinearities in equivariant DenseNets.

```bash
pip install -e "../../[dev]"
python train.py --model bispectrum --group c8 --data_dir ./pcam_data
./run_matched_sweep.sh  # full Pareto sweep: 5 models x 5 growth rates x 3 seeds
```
