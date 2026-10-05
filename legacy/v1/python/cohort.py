from __future__ import annotations

import math
import os
import random
import sys

def solve(A, b):
    n = len(b)
    L = [0.0] * (n * n)
    for i in range(n):
        s = A[i * n + i]
        for k in range(i):
            s -= L[i * n + k] * L[i * n + k]
        if s <= 1e-18:
            raise RuntimeError("normal equations were not positive definite")
        L[i * n + i] = math.sqrt(s)
        for j in range(i + 1, n):
            t = A[j * n + i]
            for k in range(i):
                t -= L[j * n + k] * L[i * n + k]
            L[j * n + i] = t / L[i * n + i]
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i * n + k] * y[k]
        y[i] = s / L[i * n + i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k * n + i] * x[k]
        x[i] = s / L[i * n + i]
    return x

def ridge(xs, ys, lam):
    if not xs:
        return []
    d = len(xs[0])
    G = [0.0] * (d * d)
    b = [0.0] * d
    for x, y in zip(xs, ys):
        for i in range(d):
            b[i] += x[i] * y
            for j in range(d):
                G[i * d + j] += x[i] * x[j]
    for i in range(d):
        G[i * d + i] += lam
    return solve(G, b)

def dot(w, x):
    return sum(a * b for a, b in zip(w, x))

def default_cohort_path():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, "..", "data", "e1_qt_cohort.csv"),
        "rasi/data/e1_qt_cohort.csv",
        "data/e1_qt_cohort.csv",
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]

class Cohort:
    def __init__(self, path):
        rows = []
        with open(path) as f:
            header = [h.strip() for h in f.readline().strip().split(",")]
            self.columns = header
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").rstrip("\r").split(",")
                rows.append(dict(zip(header, parts)))
        if not rows:
            raise SystemExit(f"empty cohort {path}")
        need = ("participant", "drug", "time", "contrast")
        for col in need:
            if col not in header:
                raise SystemExit(f"cohort {path} is missing a {col} column")
        self.path = path
        self.has_concentration = "concentration" in header
        self.has_baseline = "baseline" in header
        self.ids = sorted({r["participant"] for r in rows})
        self.drugs = sorted({r["drug"] for r in rows})
        self.times = sorted({float(r["time"]) for r in rows})
        self.nP, self.nA, self.nT = len(self.ids), len(self.drugs), len(self.times)
        self.ip = {s: i for i, s in enumerate(self.ids)}
        self.ia = {s: i for i, s in enumerate(self.drugs)}
        self.it = {t: i for i, t in enumerate(self.times)}
        n = self.nP * self.nA * self.nT
        self.y = [0.0] * n
        self.conc = [0.0] * n
        self.base = [0.0] * n
        self.seen = [0] * n
        self.present = [0] * (self.nP * self.nA)
        for r in rows:
            p, a, t = self.ip[r["participant"]], self.ia[r["drug"]], self.it[float(r["time"])]
            k = self.at(p, a, t)
            self.y[k] = float(r["contrast"])
            self.seen[k] = 1
            if self.has_concentration and r.get("concentration", "") != "":
                self.conc[k] = float(r["concentration"])
            if self.has_baseline and r.get("baseline", "") != "":
                self.base[k] = float(r["baseline"])
            self.present[p * self.nA + a] = 1
        self.pairs = [(p, a) for a in range(self.nA) for p in range(self.nP) if self.present[p * self.nA + a]]
        self.i2 = min(range(self.nT), key=lambda i: abs(self.times[i] - 2.0))
        qt = any("Dofetilide" in d for d in self.drugs)
        self.floor = 5.42 if qt else 1.0
        self.thresh = 10.0 if qt else 2.0
        rng = random.Random(20261001)
        self.train, self.audit, self.hid = [], [], []
        for a in range(self.nA):
            ps = [p for p, aa in self.pairs if aa == a]
            rng.shuffle(ps)
            for i, p in enumerate(ps):
                slot = self.hid if i % 5 == 0 else self.audit if i % 5 == 1 else self.train
                slot.append((p, a))
        self.train_set = set(self.train)

    def at(self, p, a, t):
        return (p * self.nA + a) * self.nT + t

    def traj(self, p, a):
        return [self.y[self.at(p, a, t)] for t in range(self.nT)]

    def logc(self, p, a, t):
        return math.log1p(max(self.conc[self.at(p, a, t)], 0.0))

def mae(cohort, pred, pairs):
    if not pairs:
        return 0.0
    s = 0.0
    n = 0
    for p, a in pairs:
        for t in range(cohort.nT):
            s += abs(pred(p, a, t) - cohort.y[cohort.at(p, a, t)])
            n += 1
    return s / n

def pair_maes(cohort, pred, pairs):
    out = []
    for p, a in pairs:
        s = 0.0
        for t in range(cohort.nT):
            s += abs(pred(p, a, t) - cohort.y[cohort.at(p, a, t)])
        out.append(s / cohort.nT)
    return out

def gain_se(cohort, base, cand, pairs):
    if not pairs:
        return 0.0, 0.0
    d = [b - c for b, c in zip(pair_maes(cohort, base, pairs), pair_maes(cohort, cand, pairs))]
    m = sum(d) / len(d)
    if len(d) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    return m, math.sqrt(var / len(d))

def paint(on, code, text):
    if not on:
        return text
    return f"\033[{code}m{text}\033[0m"

def tty(stream=None):
    return (stream or sys.stdout).isatty()

def banner(title):
    on = tty()
    print()
    print(paint(on, "1", title))

def kv(key, value):
    print(f"  {key:<12}{value}")

def rule():
    print(paint(tty(), "2", "  " + "─" * 72))
