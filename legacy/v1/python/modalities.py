from __future__ import annotations

import argparse
import json
import math
import sys

from cohort import Cohort, banner, default_cohort_path, dot, gain_se, kv, mae, paint, ridge, rule, tty

FAMILIES = {
    "statistical": ["m0", "m1", "m2", "bayes", "cf_gp"],
    "causal sequence": ["rmsn", "crn", "gnet", "causal_transformer", "te_cde"],
    "latent": ["gru", "rssm", "latent_ode", "jepa", "diffusion"],
    "foundation": ["event_transformer", "llm_lora"],
    "structural": ["dscm", "neural_sde", "ssm", "tabpfn", "retrieval", "ensemble"],
    "static": ["sx", "dr", "causal_forest", "bcf", "tarnet"],
}
BLURB = {
    "m0": "drug-by-time mean",
    "m1": "log concentration line, participant offset",
    "m2": "ridge participant-by-drug term",
    "bayes": "same ridge, tighter prior",
    "cf_gp": "drug-time mean plus a borrow from the person's other drugs",
    "rmsn": "weighted sequence on concentration and the last contrast",
    "crn": "history only; current concentration stays out of the decoder",
    "gnet": "drug curve, quadratic in time",
    "causal_transformer": "attention over past contrasts, or similar patients if the arm is new",
    "te_cde": "drift of the drug-time mean",
    "gru": "blend of the last contrast and the concentration line",
    "rssm": "slow average of the ridge prediction",
    "latent_ode": "step from the last contrast toward the concentration line",
    "jepa": "early-time summary carried on the concentration curve",
    "diffusion": "smoothed posterior mean",
    "event_transformer": "other drugs of the same person at this time",
    "llm_lora": "rank-1 time factor inside each drug",
    "dscm": "baseline plus a drug-time effect",
    "neural_sde": "drift of the ridge mean",
    "ssm": "moving average of the concentration line",
    "tabpfn": "neighbours in baseline, time, concentration, drug",
    "retrieval": "nearest recorded trajectories on this drug",
    "ensemble": "mean of the ridge, the concentration line, and retrieval",
    "sx": "shared ridge, drug curve, cross-drug fill",
    "dr": "concentration line plus the residual in the same baseline band",
    "causal_forest": "drug-by-time mean inside a baseline band",
    "bcf": "baseline curve plus a shrunk drug effect",
    "tarnet": "shared head averaged with a drug head",
}

def family_of(name):
    for fam, names in FAMILIES.items():
        if name in names:
            return fam
    raise SystemExit(f"unknown modality {name}")

class Fit:
    
    def __init__(self, cohort: Cohort):
        self.c = cohort
        c = cohort
        buckets = {}
        for p, a in c.train:
            for t in range(c.nT):
                buckets.setdefault((a, t), []).append(c.y[c.at(p, a, t)])
        self.mean = {k: sum(v) / len(v) for k, v in buckets.items()}
        self.m1_w = self._m1()
        self.off = self._offsets()
        self.theta, self.mu, self.beta, self.delta = self._ridge(5.0, 5.0)
        self.theta_b, _, _, _ = self._ridge(25.0, 25.0)
        self.gnet_w = self._gnet()
        self.lora_w = self._lora()
        self.prog_w = self._prognostic()
        self.level_w = self._level()
        self.dscm_w, self.base_mean = self._dscm()
        self.tert = self._tertiles()
        self.cell = self._cells()
        self.person = self._person()
        self.tau, self.gru_z, self.ssm_decay = self._rates()
        self._donors = {}
        self._seq = {}

    def drug_time(self, a, t):
        return self.mean.get((a, t), 0.0)

    def m1_raw(self, p, a, t):
        w = self.m1_w[a]
        return w[0] + w[1] * self.c.logc(p, a, t)

    def m1(self, p, a, t):
        return self.m1_raw(p, a, t) + self.off.get(p, 0.0)

    def lin(self, th, p, a, t):
        return th[self.mu(a, t)] + th[self.beta(p)] + th[self.delta(p, a)]

    def m2(self, p, a, t):
        return self.lin(self.theta, p, a, t)

    def bayes(self, p, a, t):
        return self.lin(self.theta_b, p, a, t)

    def _m1(self):
        c = self.c
        out = {}
        for a in range(c.nA):
            xs, ys = [], []
            for p, aa in c.train:
                if aa != a:
                    continue
                for t in range(c.nT):
                    xs.append([1.0, c.logc(p, a, t)])
                    ys.append(c.y[c.at(p, a, t)])
            out[a] = ridge(xs, ys, 1.0) if xs else [0.0, 0.0]
        return out

    def _offsets(self):
        c = self.c
        acc = {p: [] for p in range(c.nP)}
        for p, a in c.train:
            for t in range(c.nT):
                acc[p].append(c.y[c.at(p, a, t)] - self.m1_raw(p, a, t))
        return {p: (sum(v) / len(v) if v else 0.0) for p, v in acc.items()}

    def _ridge(self, lam_b, lam_d):
        c = self.c
        P = c.nA * c.nT + c.nP + c.nP * c.nA
        G = [0.0] * (P * P)
        b = [0.0] * P

        def mu(a, t):
            return a * c.nT + t

        def beta(p):
            return c.nA * c.nT + p

        def delta(p, a):
            return c.nA * c.nT + c.nP + (p * c.nA + a)

        for p, a in c.train:
            for t in range(c.nT):
                ix = (mu(a, t), beta(p), delta(p, a))
                y = c.y[c.at(p, a, t)]
                for u in range(3):
                    b[ix[u]] += y
                    G[ix[u] * P + ix[u]] += 1
                    for v in range(u + 1, 3):
                        G[ix[u] * P + ix[v]] += 1
                        G[ix[v] * P + ix[u]] += 1
        for i in range(c.nA * c.nT):
            G[i * P + i] += 1e-4
        for p in range(c.nP):
            G[beta(p) * P + beta(p)] += lam_b
        for p in range(c.nP):
            for a in range(c.nA):
                d = delta(p, a)
                G[d * P + d] += lam_d
        from cohort import solve

        return solve(G, b), mu, beta, delta

    def _gnet(self):
        c = self.c
        out = {}
        for a in range(c.nA):
            xs, ys = [], []
            for p, aa in c.train:
                if aa != a:
                    continue
                for t in range(c.nT):
                    tm = c.times[t]
                    xs.append([1.0, tm, tm * tm, c.logc(p, a, t)])
                    ys.append(c.y[c.at(p, a, t)])
            out[a] = ridge(xs, ys, 1.0) if xs else [0.0, 0.0, 0.0, 0.0]
        return out

    def gnet(self, p, a, t):
        tm = self.c.times[t]
        return dot(self.gnet_w[a], [1.0, tm, tm * tm, self.c.logc(p, a, t)])

    def _lora(self):
        c = self.c
        out = {}
        for a in range(c.nA):
            xs, ys = [], []
            for p, aa in c.train:
                if aa != a:
                    continue
                for t in range(c.nT):
                    factor = math.exp(-0.5 * ((c.times[t] - 2.0) / 4.0) ** 2)
                    xs.append([1.0, factor])
                    ys.append(c.y[c.at(p, a, t)])
            out[a] = ridge(xs, ys, 1.0) if xs else [0.0, 0.0]
        return out

    def lora(self, p, a, t):
        factor = math.exp(-0.5 * ((self.c.times[t] - 2.0) / 4.0) ** 2)
        return dot(self.lora_w[a], [1.0, factor])

    def _prognostic(self):
        c = self.c
        xs, ys = [], []
        for p, a in c.train:
            for t in range(c.nT):
                xs.append([1.0, c.times[t], c.logc(p, a, t), c.base[c.at(p, a, t)] / 400.0])
                ys.append(c.y[c.at(p, a, t)])
        return ridge(xs, ys, 1.0) if xs else [0.0, 0.0, 0.0, 0.0]

    def prognostic(self, p, a, t):
        c = self.c
        return dot(self.prog_w, [1.0, c.times[t], c.logc(p, a, t), c.base[c.at(p, a, t)] / 400.0])

    def _level(self):
        c = self.c
        xs, ys = [], []
        for p, a in c.train:
            for t in range(c.nT):
                xs.append([1.0, c.times[t], c.base[c.at(p, a, t)] / 400.0])
                ys.append(c.y[c.at(p, a, t)])
        return ridge(xs, ys, 1.0) if xs else [0.0, 0.0, 0.0]

    def level(self, p, a, t):
        c = self.c
        return dot(self.level_w, [1.0, c.times[t], c.base[c.at(p, a, t)] / 400.0])

    def _person(self):
        c = self.c
        acc = {p: [] for p in range(c.nP)}
        for p, a in c.train:
            for t in range(c.nT):
                acc[p].append(c.y[c.at(p, a, t)] - self.drug_time(a, t))
        return {p: (sum(v) / len(v) if v else 0.0) for p, v in acc.items()}

    def _dscm(self):
        c = self.c
        bases = [c.base[c.at(p, a, 0)] for p, a in c.train]
        base_mean = sum(bases) / len(bases) if bases else 0.0
        xs, ys = [], []
        for p, a in c.train:
            for t in range(c.nT):
                xs.append([1.0, c.base[c.at(p, a, t)] - base_mean, self.drug_time(a, t)])
                ys.append(c.y[c.at(p, a, t)])
        w = ridge(xs, ys, 1.0) if xs else [0.0, 0.0, 1.0]
        return w, base_mean

    def _tertiles(self):
        c = self.c
        bases = sorted(c.base[c.at(p, a, 0)] for p, a in c.train)
        if not bases:
            return [0.0, 0.0]
        return [bases[len(bases) // 3], bases[(2 * len(bases)) // 3]]

    def leaf(self, p, a):
        b = self.c.base[self.c.at(p, a, 0)]
        lo, hi = self.tert
        return 0 if b < lo else 1 if b < hi else 2

    def _cells(self):
        c = self.c
        buckets = {}
        for p, a in c.train:
            leaf = self.leaf(p, a)
            for t in range(c.nT):
                buckets.setdefault((leaf, a, t), []).append(c.y[c.at(p, a, t)] - self.m1_raw(p, a, t))
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    def _train_mae(self, fn):
        return mae(self.c, fn, self.c.train)

    def _rates(self):
        c = self.c

        def drift(tau, target):
            def fn(p, a, t):
                pred = target(p, a, 0)
                for tt in range(1, t + 1):
                    dt = max(c.times[tt] - c.times[tt - 1], 0.25)
                    pred = pred + (1 - math.exp(-dt / tau)) * (target(p, a, tt) - pred)
                return pred

            return fn

        tau = min((1.0, 2.0, 4.0, 8.0, 12.0), key=lambda z: self._train_mae(drift(z, self._target_m0)))
        z = min((0.2, 0.45, 0.7), key=lambda g: self._train_mae(self._gru_factory(g)))
        decay = min((0.35, 0.55, 0.75), key=lambda d: self._train_mae(self._ssm_factory(d)))
        return tau, z, decay

    def _target_m0(self, p, a, t):
        return self.drug_time(a, t)

    def _gru_factory(self, z):
        def fn(p, a, t):
            pred = self.m1(p, a, 0)
            for tt in range(1, t + 1):
                pred = z * pred + (1 - z) * self.m1(p, a, tt)
            return pred

        return fn

    def _ssm_factory(self, decay):
        def fn(p, a, t):
            pred = self.m1(p, a, 0)
            for tt in range(1, t + 1):
                pred = decay * pred + (1 - decay) * self.m1(p, a, tt)
            return pred

        return fn

    def cf_gp(self, p, a, t):
        c = self.c
        num, den = 0.0, 0.0
        for aa in range(c.nA):
            if aa == a or (p, aa) not in c.train_set:
                continue
            for tt in range(c.nT):
                w = math.exp(-0.5 * ((c.times[t] - c.times[tt]) / 6.0) ** 2)
                num += w * (c.y[c.at(p, aa, tt)] - self.drug_time(aa, tt))
                den += w
        return self.drug_time(a, t) + (num / den if den else 0.0)

    def donors(self, p, a):
        key = (p, a)
        if key in self._donors:
            return self._donors[key]
        c = self.c
        ranked = []
        for q, aa in c.train:
            if aa != a or q == p:
                continue
            common = [
                d
                for d in range(c.nA)
                if d != a and (p, d) in c.train_set and (q, d) in c.train_set
            ]
            if not common:
                dist = 1e9
            else:
                ss = 0.0
                for d in common:
                    ss += (c.y[c.at(p, d, c.i2)] - c.y[c.at(q, d, c.i2)]) ** 2
                dist = math.sqrt(ss / len(common))
            ranked.append((dist, q))
        ranked.sort()
        got = [q for _, q in ranked[:4]]
        self._donors[key] = got
        return got

    def retrieval(self, p, a, t):
        qs = self.donors(p, a)
        if not qs:
            return self.drug_time(a, t)
        c = self.c
        return sum(c.y[c.at(q, a, t)] for q in qs) / len(qs)

    def tabpfn(self, p, a, t):
        c = self.c
        b0 = c.base[c.at(p, a, 0)]
        lc = c.logc(p, a, t)
        scored = []
        for pp, aa in c.train:
            same = 0.0 if aa == a else 1.0
            for tt in range(c.nT):
                dist = (
                    ((c.base[c.at(pp, aa, 0)] - b0) / 25.0) ** 2
                    + ((c.times[tt] - c.times[t]) / 3.0) ** 2
                    + ((c.logc(pp, aa, tt) - lc) / 1.5) ** 2
                    + same
                )
                scored.append((dist, c.y[c.at(pp, aa, tt)]))
        scored.sort()
        num, den = 0.0, 0.0
        for dist, yy in scored[:12]:
            w = math.exp(-dist)
            num += w * yy
            den += w
        return num / den if den else self.drug_time(a, t)

    def walk(self, p, a, t, target, rate):
        c = self.c
        pred = target(p, a, 0)
        for tt in range(1, t + 1):
            dt = max(c.times[tt] - c.times[tt - 1], 0.25)
            pred = pred + (1 - math.exp(-dt / rate)) * (target(p, a, tt) - pred)
        return pred

    def smoothed(self, fn, p, a, t):
        c = self.c
        acc, wsum = 0.0, 0.0
        for tt in range(c.nT):
            w = math.exp(-abs(c.times[tt] - c.times[t]) / 4.0)
            acc += w * fn(p, a, tt)
            wsum += w
        return acc / wsum

def build_predictors(fit: Fit):
    c = fit.c

    def m0(p, a, t):
        return fit.drug_time(a, t)

    def m1(p, a, t):
        return fit.m1(p, a, t)

    def m2(p, a, t):
        return fit.m2(p, a, t)

    def bayes(p, a, t):
        return fit.bayes(p, a, t)

    def cf_gp(p, a, t):
        return fit.cf_gp(p, a, t)

    def rmsn(p, a, t):
        pred = m1(p, a, 0)
        for tt in range(t + 1):
            prev = pred
            level = m1(p, a, tt)
            if tt == 0:
                pred = level
            else:
                pred = 0.55 * level + 0.45 * prev
        return pred

    def crn(p, a, t):
        pred = fit.drug_time(a, 0)
        for tt in range(t + 1):
            level = fit.drug_time(a, tt)
            pred = level if tt == 0 else 0.5 * level + 0.5 * pred
        return pred

    def gnet(p, a, t):
        return fit.gnet(p, a, t)

    def causal_transformer(p, a, t):
        if (p, a) in c.train_set and t > 0:
            num, den = 0.0, 0.0
            for tt in range(t):
                w = math.exp(-(c.times[t] - c.times[tt]) / 4.0)
                num += w * c.y[c.at(p, a, tt)]
                den += w
            return 0.6 * (num / den) + 0.4 * fit.drug_time(a, t)
        num, den = 0.0, 0.0
        b0 = c.base[c.at(p, a, 0)]
        for pp, aa in c.train:
            if aa != a:
                continue
            dist = abs(c.base[c.at(pp, aa, 0)] - b0) / 20.0 + abs(c.times[min(t, c.nT - 1)] - c.times[t])
            w = math.exp(-dist)
            num += w * c.y[c.at(pp, aa, t)]
            den += w
        if den == 0:
            return fit.drug_time(a, t)
        return num / den

    def te_cde(p, a, t):
        return fit.walk(p, a, t, fit._target_m0, fit.tau)

    def gru(p, a, t):
        return fit._gru_factory(fit.gru_z)(p, a, t)

    def rssm(p, a, t):
        pred = m2(p, a, 0)
        for tt in range(1, t + 1):
            pred = 0.45 * pred + 0.55 * m2(p, a, tt)
        return pred

    def latent_ode(p, a, t):
        return 0.5 * m1(p, a, t) + 0.5 * fit.walk(p, a, t, fit.m1, fit.tau)

    def jepa(p, a, t):
        return fit.drug_time(a, t) + fit.person.get(p, 0.0)

    def diffusion(p, a, t):
        return 0.5 * bayes(p, a, t) + 0.5 * fit.smoothed(bayes, p, a, t)

    def event_transformer(p, a, t):
        num, den = 0.0, 0.0
        for aa in range(c.nA):
            if aa == a or (p, aa) not in c.train_set:
                continue
            num += c.y[c.at(p, aa, t)] - fit.drug_time(aa, t)
            den += 1.0
        return fit.drug_time(a, t) + (num / den if den else 0.0)

    def llm_lora(p, a, t):
        return fit.lora(p, a, t)

    def dscm(p, a, t):
        return dot(
            fit.dscm_w,
            [1.0, c.base[c.at(p, a, t)] - fit.base_mean, fit.drug_time(a, t)],
        )

    def neural_sde(p, a, t):
        return fit.walk(p, a, t, fit.m2, fit.tau)

    def ssm(p, a, t):
        return fit._ssm_factory(fit.ssm_decay)(p, a, t)

    def retrieval(p, a, t):
        return fit.retrieval(p, a, t)

    def ensemble(p, a, t):
        return (m0(p, a, t) + m2(p, a, t) + m1(p, a, t) + cf_gp(p, a, t) + retrieval(p, a, t)) / 5

    def sx(p, a, t):
        return (m2(p, a, t) + gnet(p, a, t) + cf_gp(p, a, t)) / 3

    def dr(p, a, t):
        return m1(p, a, t) + fit.cell.get((fit.leaf(p, a), a, t), 0.0)

    def causal_forest(p, a, t):
        leaf = fit.leaf(p, a)
        if (leaf, a, t) in fit.cell:
            leaf_mean = fit.m1_raw(p, a, t) + fit.cell[(leaf, a, t)]
            return 0.5 * leaf_mean + 0.5 * fit.drug_time(a, t)
        return fit.drug_time(a, t)

    def bcf(p, a, t):
        te = fit.drug_time(a, t) - _pooled_time(t)
        indiv = fit.person.get(p, 0.0) * 0.5
        return fit.level(p, a, t) + te + indiv

    def tarnet(p, a, t):
        return 0.5 * m2(p, a, t) + 0.5 * m1(p, a, t)

    pooled = []
    for t in range(c.nT):
        vals = [fit.drug_time(a, t) for a in range(c.nA)]
        pooled.append(sum(vals) / len(vals))

    def _pooled_time(t):
        return pooled[t]

    return {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "bayes": bayes,
        "cf_gp": cf_gp,
        "rmsn": rmsn,
        "crn": crn,
        "gnet": gnet,
        "causal_transformer": causal_transformer,
        "te_cde": te_cde,
        "gru": gru,
        "rssm": rssm,
        "latent_ode": latent_ode,
        "jepa": jepa,
        "diffusion": diffusion,
        "event_transformer": event_transformer,
        "llm_lora": llm_lora,
        "dscm": dscm,
        "neural_sde": neural_sde,
        "ssm": ssm,
        "tabpfn": fit.tabpfn,
        "retrieval": retrieval,
        "ensemble": ensemble,
        "sx": sx,
        "dr": dr,
        "causal_forest": causal_forest,
        "bcf": bcf,
        "tarnet": tarnet,
    }

def score_all(cohort, preds):
    rows = []
    for fam, names in FAMILIES.items():
        for name in names:
            fn = preds[name]
            rows.append(
                {
                    "family": fam,
                    "modality": name,
                    "blurb": BLURB[name],
                    "gate": mae(cohort, fn, cohort.audit),
                    "hidden": mae(cohort, fn, cohort.hid),
                    "train": mae(cohort, fn, cohort.train),
                }
            )
    rows.sort(key=lambda r: (r["gate"], r["modality"]))
    return rows

def stack_of(rows, preds, cohort):
    leader = rows[0]
    base = preds[leader["modality"]]
    band = [r for r in rows if r["gate"] <= leader["gate"] + 0.15]
    if len(band) > 1:
        def tied(p, a, t):
            return sum(preds[r["modality"]](p, a, t) for r in band) / len(band)

        tied_gate = mae(cohort, tied, cohort.audit)
        if tied_gate <= leader["gate"] + 1e-9:
            return {
                "modality": "stack",
                "members": [r["modality"] for r in band],
                "weights": [1.0 / len(band)] * len(band),
                "fn": tied,
                "gate": tied_gate,
                "why": "tied adapters averaged; the average did not lose on the gate",
            }
    top = rows[:4]
    weights = [1.0 / max(r["gate"], 1e-6) for r in top]
    z = sum(weights)
    weights = [w / z for w in weights]

    def ens(p, a, t):
        return sum(w * preds[r["modality"]](p, a, t) for w, r in zip(weights, top))

    delta, se = gain_se(cohort, base, ens, cohort.audit)
    if delta > se and delta > 0:
        return {
            "modality": "stack",
            "members": [r["modality"] for r in top],
            "weights": weights,
            "fn": ens,
            "gate": mae(cohort, ens, cohort.audit),
            "why": "stack beat the leading adapter by more than the gate's standard error",
        }
    return {
        "modality": leader["modality"],
        "members": [leader["modality"]],
        "weights": [1.0],
        "fn": base,
        "gate": leader["gate"],
        "why": "lowest gate error; averaging the leaders did not help the gate",
    }

def describe_cohort(cohort):
    cols = "  ".join(cohort.columns)
    kv("cohort", cohort.path)
    kv("columns", cols)
    kv(
        "size",
        f"{cohort.nP} participants    {cohort.nA} drugs    {cohort.nT} times    {len(cohort.pairs)} periods",
    )
    kv(
        "split",
        f"{len(cohort.train)} fit    {len(cohort.audit)} gate    {len(cohort.hid)} hidden    (stratified within drug)",
    )

def print_board(rows, shipped, m0_hidden):
    on = tty()
    names = set(shipped) if isinstance(shipped, (list, tuple, set)) else {shipped}
    print()
    print(paint(on, "2", f"  {'family':<18}{'modality':<22}{'gate ms':>10}{'hidden ms':>12}{'vs mean':>10}"))
    rule()
    for r in rows:
        vs = r["hidden"] - m0_hidden
        mark = "   ship" if r["modality"] in names else ""
        line = f"  {r['family']:<18}{r['modality']:<22}{r['gate']:10.2f}{r['hidden']:12.2f}{vs:10.2f}{mark}"
        if r["modality"] in names:
            line = paint(on, "1;32", line)
        print(line)
    print()

def print_decision(ship, rows, cohort, preds):
    on = tty()
    m0_h = next(r["hidden"] for r in rows if r["modality"] == "m0")
    hidden = mae(cohort, ship["fn"], cohort.hid)
    gain = m0_h - hidden
    name = ship["modality"]
    blurb = BLURB.get(name, "average of " + ", ".join(ship["members"]))
    print(paint(on, "1;32", f"  ships        {name}"))
    print(f"  readout      {blurb}")
    print(f"  because      {ship['why']}")
    print(f"  gate         {ship['gate']:.2f} ms")
    tone = "32" if gain > 0 else "31"
    print(paint(on, tone, f"  hidden test  {hidden:.2f} ms     saturated mean {m0_h:.2f} ms     gain {gain:+.2f} ms"))
    print("  hidden periods were not used to fit or to choose")
    print()
    return hidden, gain

def emit(cmd, cohort, rows, ship, preds, as_json, extra=None):
    m0_h = next(r["hidden"] for r in rows if r["modality"] == "m0")
    hidden = mae(cohort, ship["fn"], cohort.hid)
    payload = {
        "cmd": cmd,
        "cohort": cohort.path,
        "columns": cohort.columns,
        "participants": cohort.nP,
        "drugs": cohort.drugs,
        "fit": len(cohort.train),
        "gate": len(cohort.audit),
        "hidden_periods": len(cohort.hid),
        "ship": ship["modality"],
        "members": ship["members"],
        "gate_ms": ship["gate"],
        "hidden_ms": hidden,
        "m0_hidden_ms": m0_h,
        "gain_ms": m0_h - hidden,
        "why": ship["why"],
        "leaderboard": [{k: r[k] for k in ("family", "modality", "gate", "hidden", "train")} for r in rows],
    }
    if extra:
        payload.update(extra)
    if as_json:
        print(json.dumps(payload))
        return payload
    label = {"train": "train", "test": "test", "apply": "apply", "infer": "infer"}[cmd]
    banner(f"RASI  {label}    modality adapters")
    describe_cohort(cohort)
    if cmd == "train":
        kv("fit on", "training periods only, concentration and baseline included where recorded")
    elif cmd == "test":
        kv("scored", "hidden periods, after the gate has already chosen")
    elif cmd == "apply":
        kv("rule", "ship the lowest gate error, or a stack that beats it by more than one SE")
    print_board(rows, ship["members"], m0_h)
    if ship["modality"] == "stack":
        on = tty()
        members = ", ".join(f"{n} {w:.2f}" for n, w in zip(ship["members"], ship["weights"]))
        print(paint(on, "1;32", f"  stack        {members}"))
    print_decision(ship, rows, cohort, preds)
    return payload

def main():
    ap = argparse.ArgumentParser(prog="modalities")
    ap.add_argument("cmd", choices=["list", "train", "test", "infer", "apply"])
    ap.add_argument("--cohort", default="")
    ap.add_argument("--participant", default="")
    ap.add_argument("--drug", default="")
    ap.add_argument("--time", type=float, default=2.0)
    ap.add_argument("--modality", default="")
    ap.add_argument("--format", choices=["auto", "json", "text"], default="auto")
    args = ap.parse_args()
    if args.cmd == "list":
        banner("RASI  modalities")
        print("  28 adapters. train, test, infer, apply.")
        for fam, names in FAMILIES.items():
            print(paint(tty(), "2", f"\n  {fam}"))
            for n in names:
                print(f"    {n:<22}{BLURB[n]}")
        print()
        return 0
    path = args.cohort or default_cohort_path()
    cohort = Cohort(path)
    fit = Fit(cohort)
    preds = build_predictors(fit)
    rows = score_all(cohort, preds)
    if args.modality:
        if args.modality not in preds:
            print(f"unknown modality {args.modality}", file=sys.stderr)
            return 2
        chosen = next(r for r in rows if r["modality"] == args.modality)
        ship = {
            "modality": args.modality,
            "members": [args.modality],
            "weights": [1.0],
            "fn": preds[args.modality],
            "gate": chosen["gate"],
            "why": "requested on the command line",
        }
    else:
        ship = stack_of(rows, preds, cohort)
    as_json = args.format == "json" or (args.format == "auto" and not tty())
    if args.cmd in ("train", "test", "apply"):
        emit(args.cmd, cohort, rows, ship, preds, as_json)
    if args.cmd == "infer":
        pid = args.participant or cohort.ids[0]
        drug = args.drug or cohort.drugs[0]
        if pid not in cohort.ip or drug not in cohort.ia:
            print("participant or drug is not in the cohort", file=sys.stderr)
            return 2
        p, a = cohort.ip[pid], cohort.ia[drug]
        t = min(range(cohort.nT), key=lambda i: abs(cohort.times[i] - args.time))
        yhat = ship["fn"](p, a, t)
        recorded = cohort.y[cohort.at(p, a, t)] if cohort.seen[cohort.at(p, a, t)] else None
        role = "fit" if (p, a) in cohort.train_set else "gate" if (p, a) in set(cohort.audit) else "hidden"
        extra = {
            "modality": ship["modality"],
            "participant": pid,
            "drug": drug,
            "time": cohort.times[t],
            "prediction": yhat,
            "concentration": cohort.conc[cohort.at(p, a, t)],
            "baseline": cohort.base[cohort.at(p, a, t)],
            "period": role,
        }
        if as_json:
            print(json.dumps(extra))
        else:
            banner("RASI  infer    modality adapters")
            kv("shipped", f"{ship['modality']}    {BLURB.get(ship['modality'], ship['why'])}")
            kv("participant", pid)
            kv("drug", f"{drug}    t={cohort.times[t]:g} h")
            kv("inputs", f"concentration {extra['concentration']:.4g}    baseline {extra['baseline']:.1f} ms")
            kv("period", role)
            print(paint(tty(), "1;32", f"  prediction   {yhat:.2f} ms"))
            if role != "fit" and recorded is not None:
                print(f"  recorded     {recorded:.2f} ms    error {abs(yhat - recorded):.2f} ms")
            print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
