#!/usr/bin/env python3
"""Build ordered job lists: jobs_nas2.txt (quantum-nas) and jobs_gpupc.txt (gpu-pc)."""
import re

repo = '/home/quantum-nas/papers/drafts/Papers/Experimento_CrossDomain_QTL'
lines = open(f'{repo}/jobs_nas.txt').read().splitlines()
lines = [l.strip() for l in lines if l.strip()]

def jobkey(l):
    m = re.match(r'--campaign (\S+) --arm (\S+) --seed (\d+)(.*)', l)
    return m.group(1), m.group(2), int(m.group(3)), m.group(4).strip()

nas_keep = []
gpupc = []
for l in lines:
    c, a, s, rest = jobkey(l)
    if c in ('scifar5', 'sfmnist5') or (c == 'smnist5' and a in ('scratch', 'synth')):
        gpupc.append(l)
    else:
        nas_keep.append((c, a, s, rest, l))

nas_order = [
    ('chain4', None),
    ('pair2', ''),
    ('pair2', '--n-layers 2 --tag L2'),
    ('pair2', '--n-layers 4 --tag L4'),
    ('pair2', '--n-qubits 6 --n-components 6 --tag q6'),
    ('pair2', '--n-qubits 8 --n-components 8 --tag q8'),
    ('pair2', '--limit-train 500 --limit-test 2000 --tag sz500'),
    ('pair2', '--limit-train 2000 --limit-test 2000 --tag sz2k'),
    ('pair2', '--limit-train 12000 --limit-test 2000 --tag sz12k'),
    ('smnist5', None),
]

nas_final = []
for c, rest in nas_order:
    for item in nas_keep:
        if item[0] != c:
            continue
        if c == 'pair2':
            if rest is None or item[3] != rest:
                continue
        nas_final.append(item[4])

# append any leftovers (safety: nothing lost)
leftover = [x[4] for x in nas_keep if x[4] not in nas_final]
nas_final += leftover

assert len(nas_final) == len(nas_keep), (len(nas_final), len(nas_keep))
open(f'{repo}/jobs_nas2.txt', 'w').write('\n'.join(nas_final) + '\n')

gp = []
for c in ('smnist5', 'sfmnist5', 'scifar5'):
    for a in ('scratch', 'synth'):
        for s in range(10):
            pat = f'--campaign {c} --arm {a} --seed {s}'
            hit = [l for l in gpupc if l == pat]
            assert len(hit) == 1, (c, a, s, hit)
            gp.append(hit[0])
open(f'{repo}/jobs_gpupc.txt', 'w').write('\n'.join(gp) + '\n')

s1, s2 = set(nas_final), set(gp)
print('nas2:', len(nas_final), 'gpupc:', len(gp))
print('union:', len(s1 | s2), 'of', len(set(lines)), 'intersect:', len(s1 & s2))
print('nas2 head:', nas_final[:3])
print('nas2 tail:', nas_final[-3:])
print('gpupc head:', gp[:2])
print('gpupc tail:', gp[-2:])
