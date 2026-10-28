| Device | Backend | PyTorch | Module | Median ms | Status |
|---|---|---|---|---:|---|
| NVIDIA A100 80GB PCIe | cuda | 2.11.0+cu130 | CnonCn | 0.158 | ok |
| NVIDIA A100 80GB PCIe | cuda | 2.11.0+cu130 | TorusOnTorus | 0.098 | ok |
| NVIDIA A100 80GB PCIe | cuda | 2.11.0+cu130 | DnonDn | 0.642 | ok |
| NVIDIA A100 80GB PCIe | cuda | 2.11.0+cu130 | SO2onDisk | 0.342 | ok |
| NVIDIA A100 80GB PCIe | cuda | 2.11.0+cu130 | SO3onS2 | 0.232 | ok |
| NVIDIA A100 80GB PCIe | cuda | 2.11.0+cu130 | OctaonOcta | 0.812 | ok |
| x86_64 | cpu | 2.11.0+cu130 | CnonCn | 0.082 | ok |
| x86_64 | cpu | 2.11.0+cu130 | TorusOnTorus | 84.874 | ok |
| x86_64 | cpu | 2.11.0+cu130 | DnonDn | 97.028 | ok |
| x86_64 | cpu | 2.11.0+cu130 | SO2onDisk | 58.405 | ok |
| x86_64 | cpu | 2.11.0+cu130 | SO3onS2 | 402.028 | ok |
| x86_64 | cpu | 2.11.0+cu130 | OctaonOcta | 239.000 | ok |
Wrote /home/johmathe/bispectrum/rebuttal/results/device_bench/combined.json
