"""Compute the COMPLEX rank of bootstrap matrix A for real vs complex signals."""
import sys
sys.path.insert(0, '/home/johmathe/bispectrum/src')
import numpy as np
from bispectrum.so3_on_s2 import _build_selective_index_map, load_cg_matrices

CG_DATA = load_cg_matrices(8)

def get_cg(l1, m1, l2, m2, l_out, m):
    if m1 + m2 != m: return 0.0
    if l_out < abs(l1-l2) or l_out > l1+l2: return 0.0
    cg = CG_DATA[(l1, l2)]
    row = (m1+l1)*(2*l2+1) + (m2+l2)
    start = 0
    for lv in range(abs(l1-l2), l1+l2+1):
        if lv == l_out: return cg[row, start+m+l_out].item()
        start += 2*lv+1
    return 0.0

def make_real_fc(lmax, seed=42):
    rng = np.random.RandomState(seed)
    fc = {}
    for l in range(lmax+1):
        c = np.zeros(2*l+1, dtype=complex)
        c[l] = rng.randn()
        for m in range(1, l+1):
            re, im = rng.randn(2)
            c[l+m] = re + 1j*im
            c[l-m] = ((-1)**m) * (re - 1j*im)
        fc[l] = c
    return fc

def make_complex_fc(lmax, seed=42):
    rng = np.random.RandomState(seed)
    fc = {}
    for l in range(lmax+1):
        c = np.zeros(2*l+1, dtype=complex)
        for m in range(-l, l+1):
            c[l+m] = rng.randn() + 1j*rng.randn()
        fc[l] = c
    return fc

def build_bootstrap_A(fc, lt, triples):
    n = 2*lt+1
    rows = []
    for l1,l2,lv in triples:
        if lv == lt and max(l1,l2) < lt:
            row = np.zeros(n, dtype=complex)
            for m in range(-lt, lt+1):
                c_m = 0j
                for m1 in range(-l1, l1+1):
                    m2 = m-m1
                    if abs(m2)>l2: continue
                    c_m += get_cg(l1,m1,l2,m2,lt,m)*fc[l1][l1+m1]*fc[l2][l2+m2]
                row[lt+m] = np.conj(c_m)
            rows.append(row)
        elif l2 == lt and lv < lt and l1 < lt:
            row = np.zeros(n, dtype=complex)
            for m2 in range(-lt, lt+1):
                d = 0j
                for m1 in range(-l1, l1+1):
                    m = m1+m2
                    if abs(m)>lv: continue
                    d += get_cg(l1,m1,lt,m2,lv,m)*fc[l1][l1+m1]*np.conj(fc[lv][lv+m])
                row[lt+m2] = d
            rows.append(row)
    return np.array(rows) if rows else np.zeros((0,n), dtype=complex)

idx = _build_selective_index_map(8)

with open('/tmp/bootstrap_rank_results.txt', 'w') as f:
    f.write("SELECTIVE bootstrap complex rank\n")
    f.write("REAL signals:\n")
    for lt in range(4,9):
        tr = [(l1,l2,lv) for l1,l2,lv in idx if max(l1,l2,lv)==lt]
        rs = []
        for s in [42,123,999]:
            A = build_bootstrap_A(make_real_fc(8,s), lt, tr)
            sv = np.linalg.svd(A, compute_uv=False)
            rs.append(int(np.sum(sv > 1e-10*sv[0])))
        f.write(f"  ell={lt}: {A.shape} ranks={rs} expected={2*lt+1}\n")

    f.write("\nCOMPLEX signals:\n")
    for lt in range(4,9):
        tr = [(l1,l2,lv) for l1,l2,lv in idx if max(l1,l2,lv)==lt]
        rs = []
        for s in [42,123,999]:
            A = build_bootstrap_A(make_complex_fc(8,s), lt, tr)
            sv = np.linalg.svd(A, compute_uv=False)
            rs.append(int(np.sum(sv > 1e-10*sv[0])))
        f.write(f"  ell={lt}: {A.shape} ranks={rs} expected={2*lt+1}\n")

    f.write("\nFULL bispectrum at REAL signals:\n")
    for lt in [4,5,6,7]:
        all_tr = []
        for l1 in range(lt+1):
            for l2 in range(l1, lt+1):
                for lv in range(abs(l1-l2), min(l1+l2,lt)+1):
                    if max(l1,l2,lv)==lt:
                        all_tr.append((l1,l2,lv))
        A = build_bootstrap_A(make_real_fc(lt,42), lt, all_tr)
        sv = np.linalg.svd(A, compute_uv=False)
        r = int(np.sum(sv>1e-10*sv[0]))
        f.write(f"  ell={lt}: {len(all_tr)} triples A={A.shape} rank={r}/{2*lt+1}\n")

    f.write("\nDONE\n")
print("Results written to /tmp/bootstrap_rank_results.txt")
