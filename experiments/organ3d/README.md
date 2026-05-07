# OrganMNIST3D Classification

3D classification on OrganMNIST3D (11 organ classes, 28x28x28 CT volumes) comparing invariant pooling strategies in octahedral-equivariant 3D ResNets.

```bash
pip install -e "../../[dev]"
python train.py --model bispectrum --data_dir ./organ3d_data
./run_sweep.sh  # full sweep: 4 models x 3 seeds
```
