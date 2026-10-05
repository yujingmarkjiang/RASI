from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys

from cohort import Cohort, banner, default_cohort_path, kv, mae, paint, tty

def solve(A, b):
    n = len(b)
    L = [0.0] * (n * n)
    ok = True
    for i in range(n):
        s = A[i * n + i]
        for k in range(i):
            s -= L[i * n + k] * L[i * n + k]
        if not s > 1e-18:
            ok = False
            break
        L[i * n + i] = math.sqrt(s)
        for j in range(i + 1, n):
            t = A[j * n + i]
            for k in range(i):
                t -= L[j * n + k] * L[i * n + k]
            L[j * n + i] = t / L[i * n + i]
    if not ok:
        raise RuntimeError("normal equations were not positive definite")
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

class Env:
    def __init__(self, doc):
        self.nP = doc["nP"]
        self.nA = doc["nA"]
        self.nT = doc["nT"]
        self.P = doc["P"]
        self.i2 = doc["i2"]
        self.floor = doc["floor"]
        self.thresh = doc["threshold"]
        self.lam_mu = doc["lam_mu"]
        self.lam_b = doc["lam_b"]
        self.lam_d = doc["lam_d"]
        self.y = doc["y"]
        self.train = [tuple(p) for p in doc["train"]]
        self.audit = [tuple(p) for p in doc["audit"]]
        self.hid = [tuple(p) for p in doc["hid"]]
        self.theta0 = doc["theta0"]

    def mu(self, a, t):
        return a * self.nT + t

    def beta(self, p):
        return self.nA * self.nT + p

    def delta(self, p, a):
        return self.nA * self.nT + self.nP + (p * self.nA + a)

    def at(self, p, a, t):
        return (p * self.nA + a) * self.nT + t

    def traj(self, p, a):
        return [self.y[self.at(p, a, t)] for t in range(self.nT)]

    def fit(self, rows):
        P = self.P
        G = [0.0] * (P * P)
        b = [0.0] * P
        for p, a, ys, w in rows:
            ix = None
            for t in range(self.nT):
                ix = (self.mu(a, t), self.beta(p), self.delta(p, a))
                for u in range(3):
                    b[ix[u]] += w * ys[t]
                    G[ix[u] * P + ix[u]] += w
                    for v in range(u + 1, 3):
                        G[ix[u] * P + ix[v]] += w
                        G[ix[v] * P + ix[u]] += w
        for i in range(self.nA * self.nT):
            G[i * P + i] += self.lam_mu
        for p in range(self.nP):
            G[self.beta(p) * P + self.beta(p)] += self.lam_b
        for p in range(self.nP):
            for a in range(self.nA):
                d = self.delta(p, a)
                G[d * P + d] += self.lam_d
        return solve(G, b)

    def pred(self, th, p, a):
        return [th[self.mu(a, t)] + th[self.beta(p)] + th[self.delta(p, a)] for t in range(self.nT)]

    def mae(self, th, pairs):
        if not pairs:
            return 0.0
        s = 0.0
        for p, a in pairs:
            pr = self.pred(th, p, a)
            s += sum(abs(pr[t] - self.y[self.at(p, a, t)]) for t in range(self.nT)) / self.nT
        return s / len(pairs)

def real_rows(env, w=1.0):
    return [(p, a, env.traj(p, a), w) for p, a in env.train]

def spread(env, th0):
    means, vars_ = [], []
    for p, a in env.train:
        res = [env.y[env.at(p, a, t)] - (th0[env.mu(a, t)] + th0[env.beta(p)]) for t in range(env.nT)]
        m = sum(res) / len(res)
        means.append(m)
        v = sum((x - m) ** 2 for x in res) / max(len(res) - 1, 1)
        vars_.append(v)
    se2 = sum(vars_) / len(vars_)
    mm = sum(means) / len(means)
    vd = sum((x - mm) ** 2 for x in means) / max(len(means) - 1, 1)
    sd2 = max(vd - se2 / env.nT, 1.0)
    return math.sqrt(max(se2, 1e-8)), math.sqrt(sd2)

def cci_of(env, hidden_series):
    return [(h - hidden_series[0]) / env.floor for h in hidden_series]

def run_m0(env, rounds):
    buckets = {}
    for p, a in env.train:
        for t in range(env.nT):
            buckets.setdefault((a, t), []).append(env.y[env.at(p, a, t)])
    mean = {k: sum(v) / len(v) for k, v in buckets.items()}

    def mae(pairs):
        s = 0.0
        for p, a in pairs:
            s += sum(abs(mean[(a, t)] - env.y[env.at(p, a, t)]) for t in range(env.nT)) / env.nT
        return s / len(pairs)

    h0 = mae(env.hid)
    return {"cci_K": 0.0, "hidden": [h0] * (rounds + 1), "kept": 1.0, "factual0": mae(env.train)}

def run_naive(env, rounds, m, seed, plausibility=False):
    rng = random.Random(seed)
    th = env.fit(real_rows(env))
    se, sd = spread(env, th)
    hidden, factual = [env.mae(th, env.hid)], [env.mae(th, env.train)]
    for _ in range(rounds):
        syn = []
        base_rows = []
        for p, a in env.hid + env.audit:
            base = env.pred(th, p, a)
            draws = []
            for _m in range(m):
                zd = rng.gauss(0, sd)
                ys = [base[t] + zd + rng.gauss(0, se) for t in range(env.nT)]
                sse = sum((ys[t] - base[t]) ** 2 for t in range(env.nT))
                draws.append((sse, ys))
            if plausibility:
                draws.sort(key=lambda z: z[0])
                draws = draws[: max(1, len(draws) // 2)]
            for _sse, ys in draws:
                syn.append((p, a, ys, 1.0))
            base_rows.append((p, a))
        th = env.fit(real_rows(env) + syn)
        hidden.append(env.mae(th, env.hid))
        factual.append(env.mae(th, env.train))
        del base_rows
    series = cci_of(env, hidden)
    return {"cci_K": series[-1], "hidden": hidden, "factual": factual, "kept": 1.0, "cci": series}

def env_from_cohort(cohort):
    doc = {
        "nP": cohort.nP,
        "nA": cohort.nA,
        "nT": cohort.nT,
        "P": cohort.nA * cohort.nT + cohort.nP + cohort.nP * cohort.nA,
        "i2": cohort.i2,
        "floor": cohort.floor,
        "threshold": cohort.thresh,
        "lam_mu": 1e-4,
        "lam_b": 5.0,
        "lam_d": 5.0,
        "y": list(cohort.y),
        "train": [list(p) for p in cohort.train],
        "audit": [list(p) for p in cohort.audit],
        "hid": [list(p) for p in cohort.hid],
        "theta0": [0.0],
    }
    env = Env(doc)
    th = env.fit(real_rows(env))
    env.theta0 = th
    return env, th

def run_tracked(env, rounds, m, seed, plausibility):
    rng = random.Random(seed)
    th = env.fit(real_rows(env))
    se, sd = spread(env, th)
    best_th, best_au, best_k = th, env.mae(th, env.audit), 0
    hidden, factual, audits = [env.mae(th, env.hid)], [env.mae(th, env.train)], [best_au]
    for step in range(rounds):
        syn = []
        for p, a in env.hid + env.audit:
            base = env.pred(th, p, a)
            draws = []
            for _m in range(m):
                zd = rng.gauss(0, sd)
                ys = [base[t] + zd + rng.gauss(0, se) for t in range(env.nT)]
                sse = sum((ys[t] - base[t]) ** 2 for t in range(env.nT))
                draws.append((sse, ys))
            if plausibility:
                draws.sort(key=lambda z: z[0])
                draws = draws[: max(1, len(draws) // 2)]
            for _sse, ys in draws:
                syn.append((p, a, ys, 1.0))
        th = env.fit(real_rows(env) + syn)
        au = env.mae(th, env.audit)
        if au < best_au:
            best_au, best_th, best_k = au, th, step + 1
        hidden.append(env.mae(th, env.hid))
        factual.append(env.mae(th, env.train))
        audits.append(au)
    series = cci_of(env, hidden)
    return {
        "cci_K": series[-1],
        "hidden": hidden,
        "factual": factual,
        "audit": audits,
        "kept": 1.0,
        "cci": series,
        "best_theta": best_th,
        "best_audit": best_au,
        "best_round": best_k,
        "final_audit": audits[-1],
    }

def m0_predict(env):
    buckets = {}
    for p, a in env.train:
        for t in range(env.nT):
            buckets.setdefault((a, t), []).append(env.y[env.at(p, a, t)])
    mean = {k: sum(v) / len(v) for k, v in buckets.items()}

    def pred(p, a, t):
        return mean[(a, t)]

    def mae_pairs(pairs):
        if not pairs:
            return 0.0
        s = 0.0
        for p, a in pairs:
            s += sum(abs(pred(p, a, t) - env.y[env.at(p, a, t)]) for t in range(env.nT)) / env.nT
        return s / len(pairs)

    return pred, mae_pairs

def fit_baselines(cohort, rounds, m, seed):
    env, theta0 = env_from_cohort(cohort)
    m0_fn, m0_mae = m0_predict(env)
    m0 = {
        "cci_K": 0.0,
        "hidden": [m0_mae(env.hid)] * (rounds + 1),
        "audit": [m0_mae(env.audit)] * (rounds + 1),
        "kept": 1.0,
        "best_audit": m0_mae(env.audit),
        "best_round": 0,
        "final_audit": m0_mae(env.audit),
        "fn_kind": "m0",
    }
    naive = run_tracked(env, rounds, m, seed, False)
    plaus = run_tracked(env, rounds, m, seed + 1, True)
    ridge = {
        "cci_K": 0.0,
        "hidden": [env.mae(theta0, env.hid)],
        "audit": [env.mae(theta0, env.audit)],
        "kept": 1.0,
        "best_audit": env.mae(theta0, env.audit),
        "best_round": 0,
        "final_audit": env.mae(theta0, env.audit),
        "best_theta": theta0,
        "fn_kind": "ridge",
    }
    return env, m0_fn, m0, naive, plaus, ridge

def _traj(env, kind, m0_fn, naive, plaus, ridge):
    if kind == "m0":
        return lambda p, a: [m0_fn(p, a, t) for t in range(env.nT)]
    th = {"naive": naive, "plausibility": plaus, "ridge": ridge}[kind]["best_theta"]
    return lambda p, a: env.pred(th, p, a)

def _beats(env, champ, cand):
    def pair(fn):
        out = []
        for p, a in env.audit:
            pr = fn(p, a)
            out.append(sum(abs(pr[t] - env.y[env.at(p, a, t)]) for t in range(env.nT)) / env.nT)
        return out

    d = [b - c for b, c in zip(pair(champ), pair(cand))]
    if not d:
        return False, 0.0, 0.0
    m = sum(d) / len(d)
    if len(d) < 2:
        return m > 0, m, 0.0
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    se = math.sqrt(var / len(d))
    return m > se, m, se

def choose_ship(env, m0_fn, m0, naive, plaus, ridge):
    specs = {
        "m0": (m0["best_audit"], 0, "saturated mean, nothing unanchored to walk"),
        "ridge": (ridge["best_audit"], 0, "round-0 ridge, before any self-training"),
        "naive": (naive["best_audit"], naive["best_round"], "self-training round that cleared the gate"),
        "plausibility": (
            plaus["best_audit"],
            plaus["best_round"],
            "plausibility round that cleared the gate",
        ),
    }
    champ = "ridge" if ridge["best_audit"] <= m0["best_audit"] else "m0"
    for name in ("naive", "plausibility"):
        ok, _delta, _se = _beats(
            env,
            _traj(env, champ, m0_fn, naive, plaus, ridge),
            _traj(env, name, m0_fn, naive, plaus, ridge),
        )
        if ok and specs[name][0] < specs[champ][0]:
            champ = name
    audit, rnd, why = specs[champ]
    if champ in ("ridge", "m0"):
        why = specs[champ][2] + "; later rounds did not clear the gate"
    board = [(n, specs[n][0], specs[n][1], specs[n][2]) for n in ("m0", "naive", "plausibility", "ridge")]
    return {"name": champ, "audit": audit, "round": rnd, "why": why, "board": board}

def predict_shipped(env, m0_fn, naive, plaus, ridge, ship, p, a, t):
    name = ship["name"]
    if name == "m0":
        return m0_fn(p, a, t)
    if name == "naive":
        th = naive["best_theta"]
    elif name == "plausibility":
        th = plaus["best_theta"]
    else:
        th = ridge["best_theta"]
    return env.pred(th, p, a)[t]

def _public(block):
    skip = {"best_theta"}
    return {k: v for k, v in block.items() if k not in skip}

def cohort_report(args):
    path = args.cohort or default_cohort_path()
    cohort = Cohort(path)
    env, m0_fn, m0, naive, plaus, ridge = fit_baselines(cohort, args.rounds, args.m, args.seed)
    ship = choose_ship(env, m0_fn, m0, naive, plaus, ridge)
    as_json = args.format == "json" or (args.format == "auto" and not tty())
    payload = {
        "adapter": "python-baselines",
        "cohort": cohort.path,
        "columns": cohort.columns,
        "participants": cohort.nP,
        "fit": len(cohort.train),
        "gate": len(cohort.audit),
        "hidden_periods": len(cohort.hid),
        "ship": ship["name"],
        "ship_round": ship["round"],
        "ship_gate_ms": ship["audit"],
        "why": ship["why"],
        "m0": _public(m0),
        "naive": _public(naive),
        "plausibility": _public(plaus),
        "ridge": _public(ridge),
    }
    if args.cmd == "infer":
        pid = args.participant or cohort.ids[0]
        drug = args.drug or cohort.drugs[0]
        if pid not in cohort.ip or drug not in cohort.ia:
            print("participant or drug is not in the cohort", file=sys.stderr)
            return 2
        p, a = cohort.ip[pid], cohort.ia[drug]
        t = min(range(cohort.nT), key=lambda i: abs(cohort.times[i] - args.time))
        yhat = predict_shipped(env, m0_fn, naive, plaus, ridge, ship, p, a, t)
        payload.update(
            {
                "participant": pid,
                "drug": drug,
                "time": cohort.times[t],
                "prediction": yhat,
                "concentration": cohort.conc[cohort.at(p, a, t)],
                "baseline": cohort.base[cohort.at(p, a, t)],
            }
        )
    if as_json:
        print(json.dumps(payload))
        return 0
    banner(f"RASI  {args.cmd}    python baselines")
    kv("cohort", cohort.path)
    kv("columns", "  ".join(cohort.columns))
    kv(
        "size",
        f"{cohort.nP} participants    {cohort.nA} drugs    {cohort.nT} times    {len(cohort.pairs)} periods",
    )
    kv(
        "split",
        f"{len(cohort.train)} fit    {len(cohort.audit)} gate    {len(cohort.hid)} hidden",
    )
    kv("loop", f"{args.rounds} rounds    {args.m} rollouts    seed {args.seed}")
    print()
    on = tty()
    print(paint(on, "2", f"  {'model':<16}{'hidden ms':>22}{'cci':>10}{'gate now':>12}{'best gate':>12}"))
    print(paint(on, "2", "  " + "─" * 72))

    def row(name, hidden, cci, final_gate, best_gate, shipped):
        h0, h1 = hidden[0], hidden[-1]
        line = f"  {name:<16}{h0:8.2f} → {h1:<8.2f}{cci:10.3f}{final_gate:12.2f}{best_gate:12.2f}"
        if shipped:
            line = paint(on, "1;32", line + "   ship")
        print(line)

    row("m0", m0["hidden"], m0["cci_K"], m0["final_audit"], m0["best_audit"], ship["name"] == "m0")
    row("naive", naive["hidden"], naive["cci_K"], naive["final_audit"], naive["best_audit"], ship["name"] == "naive")
    row(
        "plausibility",
        plaus["hidden"],
        plaus["cci_K"],
        plaus["final_audit"],
        plaus["best_audit"],
        ship["name"] == "plausibility",
    )
    row("ridge", ridge["hidden"], 0.0, ridge["final_audit"], ridge["best_audit"], ship["name"] == "ridge")
    print()
    print(paint(on, "1;32", f"  ships        {ship['name']}    round {ship['round']}"))
    print(f"  because      {ship['why']}")
    print(f"  gate         {ship['audit']:.2f} ms")
    if args.cmd == "test":
        print("  hidden is the test. the gate chooses what ships.")
    if args.cmd == "infer":
        print(paint(on, "1;32", f"  prediction   {payload['prediction']:.2f} ms"))
        print(
            f"  query        {payload['participant']}    {payload['drug']}    t={payload['time']:g} h"
            f"    concentration {payload['concentration']:.4g}    baseline {payload['baseline']:.1f}"
        )
    print()
    return 0

def legacy_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--dump-bin")
    ap.add_argument("--out")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    path = args.data
    if args.dump_bin:
        path = path or "/tmp/rasi_baseline_env.json"
        subprocess.check_call(
            [args.dump_bin, "dump", "--env", "tiny", "--seed", str(args.seed), "--out", path]
        )
    if not path:
        print("need --data or --dump-bin", file=sys.stderr)
        return 2
    doc = json.load(open(path))
    env = Env(doc)
    th = env.fit(real_rows(env))
    worst = max(abs(th[i] - env.theta0[i]) for i in range(len(th)))
    if worst > 1e-6:
        print(f"round-0 theta disagrees with the C++ engine by {worst}", file=sys.stderr)
        return 1
    report = {
        "adapter": "python-baselines",
        "theta_abs": worst,
        "m0": run_m0(env, args.rounds),
        "naive": run_naive(env, args.rounds, args.m, args.seed, False),
        "plausibility": run_naive(env, args.rounds, args.m, args.seed + 1, True),
    }
    text = json.dumps(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    on = tty(sys.stderr)
    print(file=sys.stderr)
    print(paint(on, "1", "RASI  baselines    C++ dump check"), file=sys.stderr)
    print(f"  theta gap    {worst:.2e}", file=sys.stderr)
    print(f"  {'model':<16}{'cci':>10}", file=sys.stderr)
    for name in ("m0", "naive", "plausibility"):
        print(f"  {name:<16}{report[name]['cci_K']:10.3f}", file=sys.stderr)
    print(file=sys.stderr)
    return 0

def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("train", "test", "infer", "apply"):
        ap = argparse.ArgumentParser(prog="baselines")
        ap.add_argument("cmd", choices=["train", "test", "infer", "apply"])
        ap.add_argument("--cohort", default="")
        ap.add_argument("--participant", default="")
        ap.add_argument("--drug", default="")
        ap.add_argument("--time", type=float, default=2.0)
        ap.add_argument("--rounds", type=int, default=6)
        ap.add_argument("--m", type=int, default=4)
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--format", choices=["auto", "json", "text"], default="auto")
        return cohort_report(ap.parse_args(argv))
    return legacy_main(argv)

if __name__ == "__main__":
    raise SystemExit(main())
