import itertools
import numpy as np

# ---------- algebra from the cocycle (independent implementation) ----------
def F(x, y, twist):
    if not twist:
        return 1
    p = 0
    for i in range(3):
        for j in range(i):
            p ^= ((x >> i) & 1) & ((y >> j) & 1)
    return -1 if p else 1

def S_tensor(twist):
    S = np.zeros((8, 8, 8), dtype=np.int64)
    for a in range(8):
        for r in range(8):
            S[a ^ r, a, r] = F(a, r, twist)
    return S

S_C = S_tensor(True)
S_D = S_tensor(False)

def Lbasis(S):
    # Lb[a] = 8x8 matrix of left multiplication by basis element a
    return np.stack([S[:, a, :] for a in range(8)])

LC = Lbasis(S_C)
LD = Lbasis(S_D)

# ---------- 1. cocycle identity on all 512 triples ----------
for x, y, z in itertools.product(range(8), repeat=3):
    assert F(x, y, True) * F(x ^ y, z, True) == F(y, z, True) * F(x, y ^ z, True)
print("cocycle identity: PASS (512 triples)")

# ---------- 2. algebra sanity ----------
for i, m in enumerate([1, 2, 4]):
    assert np.array_equal(LC[m] @ LC[m], np.eye(8, dtype=np.int64)), f"e{i+1}^2 != 1"
print("signature e_i^2=+1: PASS")
for m1, m2 in [(1, 2), (1, 4), (2, 4)]:
    assert np.array_equal(LC[m1] @ LC[m2], -(LC[m2] @ LC[m1]))
print("C anticommutation: PASS")
for a, b in itertools.product(range(8), repeat=2):
    assert np.array_equal(LD[a] @ LD[b], LD[b] @ LD[a])
print("D commutativity: PASS (64 pairs)")
# e1 e2 = +e12, e2 e1 = -e12  (masks 1,2 -> 3)
assert F(1, 2, True) == 1 and F(2, 1, True) == -1
print("left/right discrimination e1e2=+e12, e2e1=-e12: PASS")

# ---------- 3. regular homomorphism L_a L_b = L_{a*b}, both arms ----------
for name, Lb, twist in [("C", LC, True), ("D", LD, False)]:
    for a, b in itertools.product(range(8), repeat=2):
        target = F(a, b, twist) * Lb[a ^ b]
        assert np.array_equal(Lb[a] @ Lb[b], target), f"hom fail {name} {a},{b}"
print("regular homomorphism L_a L_b = L_(a*b): PASS (both arms, 64 pairs each)")

# ---------- 4. verify the doc's printed L_C block against the formula ----------
# doc section 4.1, rows k=0..7, cols r=0..7: entry = sign * w_{index}
doc_idx = [
    [0,1,2,3,4,5,6,7],
    [1,0,3,2,5,4,7,6],
    [2,3,0,1,6,7,4,5],
    [3,2,1,0,7,6,5,4],
    [4,5,6,7,0,1,2,3],
    [5,4,7,6,1,0,3,2],
    [6,7,4,5,2,3,0,1],
    [7,6,5,4,3,2,1,0],
]
doc_sgn = [
    [+1,+1,+1,-1,+1,-1,-1,-1],
    [+1,+1,+1,-1,+1,-1,-1,-1],
    [+1,-1,+1,+1,+1,+1,-1,+1],
    [+1,-1,+1,+1,+1,+1,-1,+1],
    [+1,-1,-1,-1,+1,+1,+1,-1],
    [+1,-1,-1,-1,+1,+1,+1,-1],
    [+1,+1,-1,+1,+1,-1,+1,+1],
    [+1,+1,-1,+1,+1,-1,+1,+1],
]
mismatch = []
for k in range(8):
    for r in range(8):
        a = k ^ r
        s = F(a, r, True)
        if doc_idx[k][r] != a or doc_sgn[k][r] != s:
            mismatch.append((k, r, doc_idx[k][r], doc_sgn[k][r], a, s))
if mismatch:
    print("printed L_C block: MISMATCHES:", mismatch)
else:
    print("printed L_C block: PASS (all 64 entries match the cocycle formula)")

# doc 4.2 L_D: entry = +w_{k^r}, verify trivially
ok = all(F(k ^ r, r, False) == 1 for k in range(8) for r in range(8))
print("printed L_D block: PASS" if ok else "printed L_D block: FAIL")

# ---------- 5. exhaustive rank enumeration, both arms ----------
def exact_rank(M):
    sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    # guard: no ambiguous singular values
    assert not np.any((sv > 1e-9) & (sv < 1e-6)), f"ambiguous sv {sv}"
    return int(np.sum(sv > 1e-6)), sv

tot = {"C": {}, "D": {}}
cond = {"C": {}, "D": {}}
mult_ok = True
walsh_ok = True
H = np.array([[(-1) ** bin(s & x).count("1") for x in range(8)] for s in range(8)], dtype=np.int64)

for q in itertools.product([-1, 0, 1], repeat=8):
    w = np.array(q, dtype=np.int64)
    nnz = int(np.count_nonzero(w))
    MC = np.einsum("akr,a->kr", LC, w)
    MD = np.einsum("akr,a->kr", LD, w)
    rC, svC = exact_rank(MC)
    rD, svD = exact_rank(MD)
    tot["C"][rC] = tot["C"].get(rC, 0) + 1
    tot["D"][rD] = tot["D"].get(rD, 0) + 1
    cond["C"].setdefault(nnz, {}).setdefault(rC, 0)
    cond["C"][nnz][rC] += 1
    cond["D"].setdefault(nnz, {}).setdefault(rD, 0)
    cond["D"][nnz][rD] += 1
    # doc 15.3 multiplicity claim for C blocks
    if rC == 4:
        nz = svC[svC > 1e-6]
        if not (len(nz) == 4 and np.allclose(nz, nz[0], rtol=1e-9)):
            mult_ok = False
    elif rC == 8:
        s_sorted = np.sort(svC)
        g1, g2 = s_sorted[:4], s_sorted[4:]
        if not (np.allclose(g1, g1[0], rtol=1e-9) and np.allclose(g2, g2[0], rtol=1e-9)):
            mult_ok = False
    # D rank equals number of nonzero Walsh coefficients
    if rD != int(np.count_nonzero(H @ w)):
        walsh_ok = False

print("\nC multiplicity structure (4 equal / two quadruples): " + ("PASS" if mult_ok else "FAIL"))
print("D rank == nonzero Walsh coefficients: " + ("PASS" if walsh_ok else "FAIL"))

# ---------- 6. diff against the doc's tables ----------
doc_tot_C = {0: 1, 4: 672, 8: 5888}
doc_tot_D = {0: 1, 1: 16, 2: 112, 4: 672, 5: 896, 6: 1344, 8: 3520}
doc_cond_C = {
    0: {0: 1}, 1: {8: 16}, 2: {4: 48, 8: 64}, 3: {8: 448},
    4: {4: 144, 8: 976}, 5: {8: 1792}, 6: {4: 384, 8: 1408},
    7: {8: 1024}, 8: {4: 96, 8: 160},
}
doc_cond_D = {
    0: {0: 1}, 1: {8: 16}, 2: {4: 112}, 3: {8: 448},
    4: {2: 112, 5: 896, 8: 112}, 5: {8: 1792}, 6: {4: 448, 6: 1344},
    7: {8: 1024}, 8: {1: 16, 4: 112, 8: 128},
}

print("\ncomputed totals C:", dict(sorted(tot["C"].items())))
print("computed totals D:", dict(sorted(tot["D"].items())))
print("totals C match doc 14.1:", tot["C"] == doc_tot_C)
print("totals D match doc 14.1:", tot["D"] == doc_tot_D)
print("conditioned C match doc 14.2:", cond["C"] == doc_cond_C)
print("conditioned D match doc 14.2:", cond["D"] == doc_cond_D)
if cond["C"] != doc_cond_C:
    print("  C diff:", {k: (cond['C'].get(k), doc_cond_C.get(k)) for k in range(9) if cond['C'].get(k) != doc_cond_C.get(k)})
if cond["D"] != doc_cond_D:
    print("  D diff:", {k: (cond['D'].get(k), doc_cond_D.get(k)) for k in range(9) if cond['D'].get(k) != doc_cond_D.get(k)})
print("\ngrand totals:", sum(tot["C"].values()), sum(tot["D"].values()))
