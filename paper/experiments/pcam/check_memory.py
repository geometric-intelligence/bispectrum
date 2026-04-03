#!/usr/bin/env python3
"""Dry-run 1 forward+backward step per model config to verify GPU memory fits."""
from __future__ import annotations

import gc
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from model import build_model

CONFIGS = {
    'standard':    {'gr': 12},
    'norm':        {'gr': 4},
    'gate':        {'gr': 3},
    'fourier_elu': {'gr': 4},
    'bispectrum':  {'gr': 4},
}

BATCH_SIZES = [256, 128, 64, 32]

def try_step(model_name: str, gr: int, bs: int, device: torch.device) -> tuple[bool, float]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = build_model(nonlin_type=model_name, group='c8', growth_rate=gr)
        model = model.to(device)
        model.train()

        x = torch.randn(bs, 3, 96, 96, device=device)
        labels = torch.randint(0, 2, (bs,), device=device, dtype=torch.float32)

        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

        loss.backward()

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        del model, x, labels, logits, loss
        torch.cuda.empty_cache()
        gc.collect()
        return True, peak_gb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0.0


def main():
    device = torch.device('cuda')
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)}, {total_gb:.1f} GB\n')

    print(f'{"Model":<15} {"GR":>3} {"BS":>4} {"Status":<6} {"Peak GB":>8}')
    print('-' * 50)

    recommendations: dict[str, int] = {}

    for model_name, cfg in CONFIGS.items():
        gr = cfg['gr']
        found = False
        for bs in BATCH_SIZES:
            ok, peak = try_step(model_name, gr, bs, device)
            status = 'OK' if ok else 'OOM'
            peak_str = f'{peak:.2f}' if ok else '—'
            print(f'{model_name:<15} {gr:>3} {bs:>4}  {status:<6} {peak_str:>8}')
            if ok and not found:
                headroom = total_gb - peak
                if headroom > 5.0:
                    recommendations[model_name] = bs
                    found = True
                    break
                else:
                    continue
            elif ok and found:
                break

        if not found:
            for bs in BATCH_SIZES:
                ok, peak = try_step(model_name, gr, bs, device)
                if ok:
                    recommendations[model_name] = bs
                    break

    print('\n' + '=' * 50)
    print('RECOMMENDED BATCH SIZES (with >5 GB headroom):')
    print('=' * 50)
    for model_name, bs in recommendations.items():
        gr = CONFIGS[model_name]['gr']
        print(f'  {model_name:<15} gr={gr}  bs={bs}')

    print('\nSweep script batch_size_for() should use:')
    print('batch_size_for() {')
    print('    local model=$1 gr=$2')
    for model_name, bs in recommendations.items():
        print(f'    # {model_name}: bs={bs}')
    print('}')


if __name__ == '__main__':
    main()
