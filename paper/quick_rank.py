import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from bispectrum.so3_on_s2 import _build_selective_index_map, load_cg_matrices

CG = load_cg_matrices(8)
out = []

def cg(l1,m1,l2,m2,l,m):
    if m1+m2!=m or l<abs(l1-l2) or l>l1+l2: return 0.
    c=CG[(l1,l2)]; r=(m1+l1)*(2*l2+1)+(m2+l2); s=0
    for v in range(abs(l1-l2),l1+l2+1):
        if v==l: return c[r,s+m+l].item()
        s+=2*v+1
    return 0.

def rfc(lmax,seed):
    rng=np.random.RandomState(seed); fc={}
    for l in range(lmax+1):
        c=np.zeros(2*l+1,dtype=complex); c[l]=rng.randn()
        for m in range(1,l+1):
            x,y=rng.randn(2); c[l+m]=x+1j*y; c[l-m]=((-1)**m)*(x-1j*y)
        fc[l]=c
    return fc

def cfc(lmax,seed):
    rng=np.random.RandomState(seed); fc={}
    for l in range(lmax+1):
        c=np.zeros(2*l+1,dtype=complex)
        for m in range(-l,l+1): c[l+m]=rng.randn()+1j*rng.randn()
        fc[l]=c
    return fc

def bA(fc,lt,tr):
    n=2*lt+1; rows=[]
    for l1,l2,lv in tr:
        if lv==lt and max(l1,l2)<lt:
            row=np.zeros(n,dtype=complex)
            for m in range(-lt,lt+1):
                s=0j
                for m1 in range(-l1,l1+1):
                    m2=m-m1
                    if abs(m2)>l2: continue
                    s+=cg(l1,m1,l2,m2,lt,m)*fc[l1][l1+m1]*fc[l2][l2+m2]
                row[lt+m]=np.conj(s)
            rows.append(row)
        elif l2==lt and lv<lt and l1<lt:
            row=np.zeros(n,dtype=complex)
            for m2 in range(-lt,lt+1):
                s=0j
                for m1 in range(-l1,l1+1):
                    m=m1+m2
                    if abs(m)>lv: continue
                    s+=cg(l1,m1,lt,m2,lv,m)*fc[l1][l1+m1]*np.conj(fc[lv][lv+m])
                row[lt+m2]=s
            rows.append(row)
    return np.array(rows) if rows else np.zeros((0,n),dtype=complex)

idx=_build_selective_index_map(8)
outf=os.path.join(os.path.dirname(__file__), 'rank_results.txt')
with open(outf,'w') as f:
    for label,mkfc in [("REAL",rfc),("COMPLEX",cfc)]:
        f.write(f"{label} signals:\n")
        for lt in range(4,9):
            tr=[(a,b,c) for a,b,c in idx if max(a,b,c)==lt]
            rs=[]
            for s in [42,123,999]:
                A=bA(mkfc(8,s),lt,tr)
                sv=np.linalg.svd(A,compute_uv=False)
                rs.append(int(np.sum(sv>1e-10*sv[0])) if sv[0]>0 else 0)
            f.write(f"  l={lt} shape={A.shape} ranks={rs} need={2*lt+1}\n")
        f.write("\n")
    f.write("FULL bispec REAL:\n")
    for lt in [4,5,6,7]:
        at=[]
        for a in range(lt+1):
            for b in range(a,lt+1):
                for c in range(abs(a-b),min(a+b,lt)+1):
                    if max(a,b,c)==lt: at.append((a,b,c))
        A=bA(rfc(lt,42),lt,at)
        sv=np.linalg.svd(A,compute_uv=False)
        r=int(np.sum(sv>1e-10*sv[0])) if sv[0]>0 else 0
        f.write(f"  l={lt} {len(at)}triples A={A.shape} rank={r}/{2*lt+1}\n")
