# OrganMNIST3D Classification

3D classification on OrganMNIST3D (11 organ classes, 28x28x28 CT volumes) comparing invariant pooling strategies in octahedral-equivariant 3D ResNets.

```bash
pip install -e "../../[dev]"
python train.py --model bispectrum --data_dir ./organ3d_data
./run_sweep.sh              # main sweep: 4 models x 3 seeds (channels 4,8)
./run_wider_multiseed.sh    # accuracy-vs-params curves (channels 8,16 and 16,32)
./run_dataeff_multiseed.sh  # accuracy-vs-train-size curves
```
