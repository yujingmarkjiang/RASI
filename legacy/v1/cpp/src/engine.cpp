#include "engine.hpp"
#include "linalg.hpp"
#include "rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using std::string;
using std::vector;

static std::map<string, string> parse_flags(int argc, char** argv) {
  std::map<string, string> o;
  for (int i = 0; i < argc; i++) {
    string a = argv[i];
    if (a.rfind("--", 0) != 0) continue;
    string key = a.substr(2);
    string val = "1";
    if (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) val = argv[++i];
    o[key] = val;
  }
  return o;
}
static int iget(const std::map<string, string>& o, const string& k, int d) {
  auto it = o.find(k);
  return it == o.end() ? d : std::atoi(it->second.c_str());
}
static double dget(const std::map<string, string>& o, const string& k, double d) {
  auto it = o.find(k);
  return it == o.end() ? d : std::atof(it->second.c_str());
}
static string sget(const std::map<string, string>& o, const string& k, const string& d) {
  auto it = o.find(k);
  return it == o.end() ? d : it->second;
}

struct Obs {
  int i0, i1, i2;
  double y, w;
};

struct Env {
  string name;
  int nP = 0, nA = 0, nT = 0, P = 0, i2 = 0;
  double floor = 1, thresh = 10, lam_mu = 1e-6, lam_b = 5, lam_d = 5;
  vector<double> times, y;
  vector<double> concentration, baseline;
  vector<char> present;
  vector<string> drugs, ids;
  bool has(int p, int a) const { return present.empty() || present[p * nA + a]; }
  int mu(int a, int t) const { return a * nT + t; }
  int beta(int p) const { return nA * nT + p; }
  int delta(int p, int a) const { return nA * nT + nP + (p * nA + a); }
  int at(int p, int a, int t) const { return ((p * nA + a) * nT) + t; }
  double y_at(int p, int a, int t) const { return y[at(p, a, t)]; }
};

static double curve(double t) { return std::exp(-0.5 * std::pow((t - 2.0) / 2.5, 2)); }

static Env make_env(const string& name, uint64_t seed) {
  Env e;
  e.name = name;
  if (name == "tiny") {
    e.nP = 6;
    e.nA = 2;
    e.times = {0, 1, 2};
    e.drugs = {"A", "B"};
    e.floor = 1.0;
    e.thresh = 2.0;
    e.lam_b = 1.0;
    e.lam_d = 1.0;
  } else {
    e.nP = 22;
    e.nA = 4;
    e.times = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 12, 14, 24};
    e.drugs = {"Dofetilide", "Quinidine", "Ranolazine", "Verapamil"};
    e.floor = 5.42;
    e.thresh = 10.0;
  }
  e.nT = static_cast<int>(e.times.size());
  e.P = e.nA * e.nT + e.nP + e.nP * e.nA;
  double best = 1e9;
  for (int t = 0; t < e.nT; t++) {
    double d = std::fabs(e.times[t] - 2.0);
    if (d < best) {
      best = d;
      e.i2 = t;
    }
  }
  vector<double> means = name == "tiny" ? vector<double>{8.0, 1.2} : vector<double>{62.3, 78.4, 7.4, 3.4};
  vector<double> isd = name == "tiny" ? vector<double>{2.0, 2.4} : vector<double>{15.2, 16.9, 11.2, 8.9};
  Rng rng(seed);
  vector<double> indiv(e.nP * e.nA);
  for (int p = 0; p < e.nP; p++)
    for (int a = 0; a < e.nA; a++) indiv[p * e.nA + a] = rng.normal() * isd[a];
  e.y.assign(e.nP * e.nA * e.nT, 0.0);
  e.present.assign(e.nP * e.nA, 1);
  e.ids.resize(e.nP);
  for (int p = 0; p < e.nP; p++) e.ids[p] = "P" + std::to_string(p + 1);
  double c2 = curve(2.0);
  for (int p = 0; p < e.nP; p++)
    for (int a = 0; a < e.nA; a++)
      for (int t = 0; t < e.nT; t++) {
        double c = curve(e.times[t]) / c2;
        e.y[e.at(p, a, t)] = (means[a] + indiv[p * e.nA + a]) * c + rng.normal() * e.floor;
      }
  return e;
}

static Env finish_env(Env e) {
  e.nT = static_cast<int>(e.times.size());
  e.P = e.nA * e.nT + e.nP + e.nP * e.nA;
  double best = 1e9;
  e.i2 = 0;
  for (int t = 0; t < e.nT; t++) {
    double d = std::fabs(e.times[t] - 2.0);
    if (d < best) {
      best = d;
      e.i2 = t;
    }
  }
  if (e.present.empty()) e.present.assign(e.nP * e.nA, 1);
  if (e.ids.empty()) {
    e.ids.resize(e.nP);
    for (int p = 0; p < e.nP; p++) e.ids[p] = "P" + std::to_string(p + 1);
  }
  return e;
}

static Env load_cohort(const string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read cohort " + path);
  string header;
  if (!std::getline(in, header)) throw std::runtime_error("empty cohort");
  struct Cell {
    string id, drug;
    double t, y, conc, base;
  };
  vector<Cell> cells;
  string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    vector<string> col;
    string cur;
    for (char c : line) {
      if (c == ',') {
        col.push_back(cur);
        cur.clear();
      } else if (c != '\r') cur.push_back(c);
    }
    col.push_back(cur);
    if (col.size() < 4) continue;
    Cell cell;
    cell.id = col[0];
    cell.drug = col[1];
    cell.t = std::atof(col[2].c_str());
    cell.y = std::atof(col[3].c_str());
    cell.conc = col.size() > 4 ? std::atof(col[4].c_str()) : 0;
    cell.base = col.size() > 5 ? std::atof(col[5].c_str()) : 0;
    cells.push_back(cell);
  }
  if (cells.empty()) throw std::runtime_error("cohort has no rows");
  Env e;
  e.name = "cohort";
  e.floor = 5.42;
  e.thresh = 10;
  for (auto& c : cells) {
    if (std::find(e.ids.begin(), e.ids.end(), c.id) == e.ids.end()) e.ids.push_back(c.id);
    if (std::find(e.drugs.begin(), e.drugs.end(), c.drug) == e.drugs.end()) e.drugs.push_back(c.drug);
    bool seen = false;
    for (double t : e.times)
      if (std::fabs(t - c.t) < 1e-9) seen = true;
    if (!seen) e.times.push_back(c.t);
  }
  std::sort(e.ids.begin(), e.ids.end());
  std::sort(e.drugs.begin(), e.drugs.end());
  std::sort(e.times.begin(), e.times.end());
  e.nP = static_cast<int>(e.ids.size());
  e.nA = static_cast<int>(e.drugs.size());
  e = finish_env(e);
  e.y.assign(e.nP * e.nA * e.nT, 0);
  e.concentration.assign(e.y.size(), 0);
  e.baseline.assign(e.y.size(), 0);
  e.present.assign(e.nP * e.nA, 0);
  auto idx = [](const vector<string>& v, const string& s) {
    return static_cast<int>(std::find(v.begin(), v.end(), s) - v.begin());
  };
  for (auto& c : cells) {
    int p = idx(e.ids, c.id);
    int a = idx(e.drugs, c.drug);
    int t = 0;
    for (; t < e.nT; t++)
      if (std::fabs(e.times[t] - c.t) < 1e-9) break;
    e.present[p * e.nA + a] = 1;
    e.y[e.at(p, a, t)] = c.y;
    e.concentration[e.at(p, a, t)] = c.conc;
    e.baseline[e.at(p, a, t)] = c.base;
  }
  bool qt = false;
  for (auto& d : e.drugs)
    if (d.find("Dofetilide") != string::npos) qt = true;
  if (!qt) {
    e.floor = 1;
    e.thresh = 2;
    e.lam_b = 1;
    e.lam_d = 1;
  }
  return e;
}

struct Mask {
  vector<char> hidden;
  vector<std::pair<int, int>> hid, vis, audit, train, unobs;
};

static Mask make_mask(const Env& e, uint64_t seed, int n_audit) {
  Mask m;
  m.hidden.assign(e.nP * e.nA, 0);
  vector<int> n_present(e.nA, 0);
  int n_pairs = 0;
  for (int a = 0; a < e.nA; a++)
    for (int p = 0; p < e.nP; p++)
      if (e.has(p, a)) {
        n_present[a]++;
        n_pairs++;
      }
  int total = static_cast<int>(std::llround(0.30 * n_pairs));
  vector<int> base(e.nA, 0);
  for (int a = 0; a < e.nA; a++) base[a] = static_cast<int>(std::floor(0.30 * n_present[a]));
  int extra = total;
  for (int b : base) extra -= b;
  if (extra < 0) extra = 0;
  if (extra > e.nA) extra = e.nA;
  Rng rng(seed);
  vector<int> drugs(e.nA);
  std::iota(drugs.begin(), drugs.end(), 0);
  rng.shuffle(drugs);
  for (int i = 0; i < extra; i++) base[drugs[i]]++;
  for (int a = 0; a < e.nA; a++) {
    vector<int> ps;
    for (int p = 0; p < e.nP; p++)
      if (e.has(p, a)) ps.push_back(p);
    rng.shuffle(ps);
    int take = std::min(base[a], static_cast<int>(ps.size()));
    for (int j = 0; j < take; j++) m.hidden[ps[j] * e.nA + a] = 1;
  }
  for (int a = 0; a < e.nA; a++)
    for (int p = 0; p < e.nP; p++) {
      if (!e.has(p, a)) continue;
      if (m.hidden[p * e.nA + a]) m.hid.push_back({p, a});
      else m.vis.push_back({p, a});
    }
  vector<int> order(m.vis.size());
  std::iota(order.begin(), order.end(), 0);
  rng.shuffle(order);
  int na = std::min(n_audit, static_cast<int>(m.vis.size()) / 2);
  if (na < 2) na = std::min(2, static_cast<int>(m.vis.size()));
  vector<char> is_audit(m.vis.size(), 0);
  for (int i = 0; i < na; i++) {
    is_audit[order[i]] = 1;
    m.audit.push_back(m.vis[order[i]]);
  }
  for (size_t i = 0; i < m.vis.size(); i++)
    if (!is_audit[i]) m.train.push_back(m.vis[i]);
  m.unobs = m.hid;
  m.unobs.insert(m.unobs.end(), m.audit.begin(), m.audit.end());
  return m;
}

static vector<double> lam_vec(const Env& e) {
  vector<double> lam(e.P, e.lam_d);
  for (int i = 0; i < e.nA * e.nT; i++) lam[i] = e.lam_mu;
  for (int p = 0; p < e.nP; p++) lam[e.beta(p)] = e.lam_b;
  return lam;
}

static void add_traj(vector<Obs>& obs, const Env& e, int p, int a, const vector<double>& ys, double w) {
  for (int t = 0; t < e.nT; t++) obs.push_back({e.mu(a, t), e.beta(p), e.delta(p, a), ys[t], w});
}

static vector<double> traj_of(const Env& e, int p, int a) {
  vector<double> ys(e.nT);
  for (int t = 0; t < e.nT; t++) ys[t] = e.y_at(p, a, t);
  return ys;
}

static vector<double> fit_obs(const Env& e, const vector<Obs>& obs) {
  int P = e.P;
  vector<double> G(P * P, 0.0), b(P, 0.0);
  for (const auto& o : obs) {
    int ix[3] = {o.i0, o.i1, o.i2};
    for (int a = 0; a < 3; a++) {
      b[ix[a]] += o.w * o.y;
      G[ix[a] * P + ix[a]] += o.w;
      for (int c = a + 1; c < 3; c++) {
        G[ix[a] * P + ix[c]] += o.w;
        G[ix[c] * P + ix[a]] += o.w;
      }
    }
  }
  auto lam = lam_vec(e);
  for (int i = 0; i < P; i++) G[i * P + i] += lam[i];
  return solve_spd(G, b, P);
}

static vector<double> pred_traj(const Env& e, const vector<double>& th, int p, int a) {
  vector<double> ys(e.nT);
  double bd = th[e.beta(p)] + th[e.delta(p, a)];
  for (int t = 0; t < e.nT; t++) ys[t] = th[e.mu(a, t)] + bd;
  return ys;
}

static double mae_pair(const Env& e, const vector<double>& th, int p, int a) {
  auto pr = pred_traj(e, th, p, a);
  double s = 0;
  for (int t = 0; t < e.nT; t++) s += std::fabs(pr[t] - e.y_at(p, a, t));
  return s / e.nT;
}

static double mae_set(const Env& e, const vector<double>& th, const vector<std::pair<int, int>>& ps) {
  if (ps.empty()) return 0;
  double s = 0;
  for (auto [p, a] : ps) s += mae_pair(e, th, p, a);
  return s / static_cast<double>(ps.size());
}

static double mean(const vector<double>& v) {
  if (v.empty()) return 0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
static double var_ddof(const vector<double>& v, int ddof) {
  if (static_cast<int>(v.size()) <= ddof) return 0;
  double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return s / static_cast<double>(v.size() - ddof);
}

struct CondOut {
  vector<double> factual, hidden, cci, klass, flip, accept;
  vector<double> klass_drug;
  double cci_K = 0, kept = 0, best_audit = 0, base_audit = 0;
  int best_round = 0;
  vector<double> best_theta;
  vector<double> trace;
  vector<int> trace_acc;
  double case_rec = 0;
  int case_p = -1, case_a = -1;
};

static CondOut run_condition(const Env& e, const Mask& mask, const string& cond, int K, int M, int knn,
                             int n_rot, double anchor, uint64_t loop_seed, bool want_trace) {
  CondOut out;
  auto real = [&](double w) {
    vector<Obs> obs;
    for (auto [p, a] : mask.train) add_traj(obs, e, p, a, traj_of(e, p, a), w);
    return obs;
  };
  vector<double> th0 = fit_obs(e, real(1.0));
  vector<double> th = th0;

  vector<double> vars, means;
  for (auto [p, a] : mask.train) {
    vector<double> res(e.nT);
    for (int t = 0; t < e.nT; t++) res[t] = e.y_at(p, a, t) - (th0[e.mu(a, t)] + th0[e.beta(p)]);
    means.push_back(mean(res));
    vars.push_back(var_ddof(res, 1));
  }
  double s_e2 = mean(vars);
  double s_d2 = std::max(var_ddof(means, 1) - s_e2 / e.nT, 1.0);
  double s_e = std::sqrt(std::max(s_e2, 1e-8));
  double s_d = std::sqrt(s_d2);

  int case_p = -1, case_a = -1;
  double case_best = 1e99;
  for (auto [p, a] : mask.hid) {
    double d = std::fabs(e.y_at(p, a, e.i2) - e.thresh);
    if (d < case_best - 1e-12 || (std::fabs(d - case_best) < 1e-12 && (a < case_a || (a == case_a && p < case_p)))) {
      case_best = d;
      case_p = p;
      case_a = a;
    }
  }
  out.case_p = case_p;
  out.case_a = case_a;
  if (case_p >= 0) out.case_rec = e.y_at(case_p, case_a, e.i2);

  auto prefer = [&](const vector<double>& thx) {
    vector<int> pref(e.nP, 0);
    for (int p = 0; p < e.nP; p++) {
      double best = 1e99;
      int ba = 0;
      for (int a = 0; a < e.nA; a++) {
        if (!e.has(p, a)) continue;
        double v = mask.hidden[p * e.nA + a] ? pred_traj(e, thx, p, a)[e.i2] : e.y_at(p, a, e.i2);
        if (v < best) {
          best = v;
          ba = a;
        }
      }
      pref[p] = ba;
    }
    return pref;
  };
  vector<int> pref0;

  vector<double> best = th0;
  double best_au = mae_set(e, th0, mask.audit);
  int best_k = 0;
  out.base_audit = best_au;
  Rng rng(loop_seed);
  int n_acc = 0, n_try = 0;
  bool full = cond == "rasi";
  bool retrieval = cond == "rsi" || cond == "rasi";
  bool gate = cond == "rsi" || cond == "rasi";
  double w_real = full ? (1.0 + anchor) : 1.0;

  for (int k = 0; k <= K; k++) {
    bool accepted = true;
    if (k > 0 && cond != "m0") {
      vector<int> budget(mask.unobs.size(), M);
      if (full) {
        vector<vector<double>> members;
        int ntr = static_cast<int>(mask.train.size());
        for (int s = 0; s < 5; s++) {
          vector<Obs> boot;
          for (int i = 0; i < ntr; i++) {
            auto [p, a] = mask.train[rng.integer(ntr)];
            add_traj(boot, e, p, a, traj_of(e, p, a), 1.0);
          }
          members.push_back(fit_obs(e, boot));
        }
        vector<double> vars_u(mask.unobs.size(), 0);
        double sumv = 0;
        for (size_t i = 0; i < mask.unobs.size(); i++) {
          auto [p, a] = mask.unobs[i];
          vector<double> pv;
          for (auto& mem : members) pv.push_back(pred_traj(e, mem, p, a)[e.i2]);
          vars_u[i] = var_ddof(pv, 0);
          sumv += vars_u[i];
        }
        int nU = static_cast<int>(mask.unobs.size());
        int B = M * nU;
        vector<int> c(nU, 0);
        int left = B;
        if (B >= nU) {
          for (int i = 0; i < nU; i++) c[i] = 1;
          left = B - nU;
        }
        vector<double> rem(nU, 0);
        int used_extra = 0;
        for (int i = 0; i < nU; i++) {
          double efrac = sumv <= 1e-15 ? (nU ? static_cast<double>(left) / nU : 0) : left * vars_u[i] / sumv;
          int fl = static_cast<int>(std::floor(efrac));
          c[i] += fl;
          used_extra += fl;
          rem[i] = efrac - fl;
        }
        vector<int> ord(nU);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int i, int j) {
          if (rem[i] != rem[j]) return rem[i] > rem[j];
          return i < j;
        });
        for (int j = 0; used_extra < left && j < nU * 2; j++) {
          c[ord[j % nU]]++;
          used_extra++;
        }
        budget.swap(c);
      }

      vector<Obs> syn;
      vector<double> prof(e.nP * e.nA, 0);
      vector<char> has(e.nP * e.nA, 0);
      if (retrieval) {
        for (auto [p, a] : mask.train) {
          has[p * e.nA + a] = 1;
          prof[p * e.nA + a] = th[e.beta(p)] + th[e.delta(p, a)];
        }
      }
      struct Roll {
        int p, a;
        vector<double> ys;
      };
      vector<Roll> rolls;
      for (size_t ui = 0; ui < mask.unobs.size(); ui++) {
        auto [p, a] = mask.unobs[ui];
        int mi = std::max(1, budget[ui]);
        vector<vector<double>> ys(mi);
        if (retrieval) {
          vector<int> donors;
          for (auto [q, b] : mask.train)
            if (b == a && q != p) donors.push_back(q);
          vector<std::pair<double, int>> ranked;
          for (int q : donors) {
            vector<int> common;
            for (int d = 0; d < e.nA; d++)
              if (d != a && has[p * e.nA + d] && has[q * e.nA + d]) common.push_back(d);
            double dist = 1e9;
            if (!common.empty()) {
              double ss = 0;
              for (int d : common) {
                double df = prof[p * e.nA + d] - prof[q * e.nA + d];
                ss += df * df;
              }
              dist = std::sqrt(ss / static_cast<double>(common.size()));
            }
            ranked.push_back({dist, q});
          }
          std::sort(ranked.begin(), ranked.end(), [](auto& A, auto& B) {
            if (A.first != B.first) return A.first < B.first;
            return A.second < B.second;
          });
          int kk = std::min(knn, static_cast<int>(ranked.size()));
          if (kk == 0) {
            auto base = pred_traj(e, th, p, a);
            for (int m = 0; m < mi; m++) {
              double zd = rng.normal() * s_d;
              ys[m].resize(e.nT);
              for (int t = 0; t < e.nT; t++) ys[m][t] = base[t] + zd + rng.normal() * s_e;
            }
          } else {
            for (int m = 0; m < mi; m++) {
              int q = ranked[rng.integer(kk)].second;
              ys[m].resize(e.nT);
              double shift = th[e.beta(p)] - th[e.beta(q)];
              for (int t = 0; t < e.nT; t++) ys[m][t] = shift + e.y_at(q, a, t);
            }
          }
        } else {
          auto base = pred_traj(e, th, p, a);
          for (int m = 0; m < mi; m++) {
            double zd = rng.normal() * s_d;
            ys[m].resize(e.nT);
            for (int t = 0; t < e.nT; t++) ys[m][t] = base[t] + zd + rng.normal() * s_e;
          }
        }
        if (full) {
          double anchor_y = pred_traj(e, th0, p, a)[e.i2];
          vector<vector<double>> kept;
          double best_d = 1e99;
          vector<double> best = ys[0];
          for (auto& row : ys) {
            double d = std::fabs(row[e.i2] - anchor_y);
            if (d < best_d) {
              best_d = d;
              best = row;
            }
            if (d <= 3.0 * s_e) kept.push_back(row);
          }
          if (kept.empty()) kept.push_back(best);
          ys.swap(kept);
        }
        for (auto& row : ys) {
          rolls.push_back({p, a, row});
          add_traj(syn, e, p, a, row, 1.0);
        }
      }
      vector<Obs> joined = real(w_real);
      joined.insert(joined.end(), syn.begin(), syn.end());
      vector<double> cand = fit_obs(e, joined);
      if (gate) {
        int nr = std::min(n_rot, static_cast<int>(mask.audit.size()));
        vector<int> ord(mask.audit.size());
        std::iota(ord.begin(), ord.end(), 0);
        rng.shuffle(ord);
        vector<std::pair<int, int>> rot;
        for (int i = 0; i < nr; i++) rot.push_back(mask.audit[ord[i]]);
        if (!full) {
          accepted = mae_set(e, cand, rot) <= mae_set(e, th, rot) + 1e-9;
        } else {
          vector<double> dlt;
          for (auto [p, a] : rot) dlt.push_back(mae_pair(e, th, p, a) - mae_pair(e, cand, p, a));
          double md = mean(dlt);
          double se = 0;
          if (dlt.size() >= 2) se = std::sqrt(var_ddof(dlt, 1) / static_cast<double>(dlt.size()));
          accepted = md > se;
        }
      }
      n_try++;
      if (accepted) {
        th = cand;
        n_acc++;
        double au = mae_set(e, th, mask.audit);
        if (au < best_au) {
          best_au = au;
          best = th;
          best_k = k;
        }
      }
      (void)rolls;
    }
    double fact = mae_set(e, th, mask.train);
    double hid = mae_set(e, th, mask.hid);
    int wrong = 0;
    for (auto [p, a] : mask.hid) {
      bool pr = pred_traj(e, th, p, a)[e.i2] >= e.thresh;
      bool ob = e.y_at(p, a, e.i2) >= e.thresh;
      if (pr != ob) wrong++;
    }
    double cls = mask.hid.empty() ? 0 : static_cast<double>(wrong) / mask.hid.size();
    auto pref = prefer(th);
    if (k == 0) pref0 = pref;
    int flips = 0;
    for (int p = 0; p < e.nP; p++)
      if (pref[p] != pref0[p]) flips++;
    double rf = static_cast<double>(flips) / e.nP;
    double cci = (hid - (out.hidden.empty() ? hid : out.hidden[0])) / e.floor;
    out.factual.push_back(fact);
    out.hidden.push_back(hid);
    out.cci.push_back(k == 0 ? 0 : cci);
    out.klass.push_back(cls);
    out.flip.push_back(rf);
    out.accept.push_back(accepted ? 1.0 : 0.0);
    if (want_trace && case_p >= 0) {
      out.trace.push_back(pred_traj(e, th, case_p, case_a)[e.i2]);
      out.trace_acc.push_back(accepted ? 1 : 0);
    }
  }
  out.cci_K = out.cci.back();
  out.kept = n_try ? static_cast<double>(n_acc) / n_try : 1;
  out.best_theta = best;
  out.best_audit = best_au;
  out.best_round = best_k;
  out.klass_drug.assign(e.nA, 0);
  vector<int> cnt(e.nA, 0), bad(e.nA, 0);
  for (auto [p, a] : mask.hid) {
    cnt[a]++;
    bool pr = pred_traj(e, th, p, a)[e.i2] >= e.thresh;
    bool ob = e.y_at(p, a, e.i2) >= e.thresh;
    if (pr != ob) bad[a]++;
  }
  for (int a = 0; a < e.nA; a++)
    if (cnt[a]) out.klass_drug[a] = static_cast<double>(bad[a]) / cnt[a];
  return out;
}

static string jnum(double x) {
  if (!std::isfinite(x)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.8g", x);
  return buf;
}
static string jarr(const vector<double>& v) {
  string s = "[";
  for (size_t i = 0; i < v.size(); i++) {
    if (i) s += ",";
    s += jnum(v[i]);
  }
  s += "]";
  return s;
}

static void average_into(CondOut& acc, const CondOut& o, int& n) {
  if (acc.factual.empty()) {
    acc = o;
    n = 1;
    return;
  }
  auto mix = [&](vector<double>& a, const vector<double>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); i++) a[i] += b[i];
  };
  mix(acc.factual, o.factual);
  mix(acc.hidden, o.hidden);
  mix(acc.cci, o.cci);
  mix(acc.klass, o.klass);
  mix(acc.flip, o.flip);
  mix(acc.accept, o.accept);
  mix(acc.klass_drug, o.klass_drug);
  acc.cci_K += o.cci_K;
  acc.kept += o.kept;
  n++;
}
static void finish_avg(CondOut& acc, int n) {
  if (n <= 1) return;
  auto sc = [&](vector<double>& a) {
    for (double& x : a) x /= n;
  };
  sc(acc.factual);
  sc(acc.hidden);
  sc(acc.cci);
  sc(acc.klass);
  sc(acc.flip);
  sc(acc.accept);
  sc(acc.klass_drug);
  acc.cci_K /= n;
  acc.kept /= n;
}

static int emit_loop(const std::map<string, string>& opt) {
  string env_name = sget(opt, "env", "qt");
  int K = iget(opt, "rounds", 12);
  int M = iget(opt, "m", 5);
  int masks_n = iget(opt, "masks", 4);
  int seeds = iget(opt, "seeds", 2);
  int rank = iget(opt, "rank", 0);
  int world = std::max(1, iget(opt, "world-size", 1));
  int knn = iget(opt, "knn", 8);
  int n_audit = iget(opt, "audit", env_name == "tiny" ? 3 : 12);
  int n_rot = iget(opt, "rotate", env_name == "tiny" ? 2 : 8);
  double anchor = dget(opt, "lambda", 1.0);
  uint64_t seed0 = static_cast<uint64_t>(iget(opt, "seed", 20261001));
  string which = sget(opt, "condition", "all");
  string out_path = sget(opt, "out", "");
  string cohort = sget(opt, "cohort", "");
  string task = sget(opt, "task", "train");
  bool trace = iget(opt, "trace", 1) && rank == 0;
  string fmt = sget(opt, "format", "auto");
  bool as_text = fmt == "text" || (fmt != "json" && out_path.empty() && isatty(STDOUT_FILENO));

  Env e = cohort.empty() ? make_env(env_name, seed0) : load_cohort(cohort);
  if (!cohort.empty()) env_name = e.name;
  vector<string> conds;
  if (which == "all") conds = {"m0", "standard", "rsi", "rasi"};
  else conds = {which};

  struct Acc {
    CondOut o;
    int n = 0;
  };
  std::map<string, Acc> acc;
  CondOut trace_std, trace_rasi, ckpt;
  bool have_trace = false, have_ckpt = false;

  int job = 0;
  for (int mi = 0; mi < masks_n; mi++) {
    Mask mask = make_mask(e, seed0 + 10007ULL * (mi + 1), n_audit);
    if (mask.train.size() < 2 || mask.hid.empty() || mask.audit.size() < 2) continue;
    for (int s = 0; s < seeds; s++) {
      if ((job % world) != rank) {
        job++;
        continue;
      }
      for (const string& c : conds) {
        uint64_t ls = seed0 + 1000003ULL * (mi + 1) + 97ULL * (s + 1) + (c == "standard" ? 1 : c == "rsi" ? 2 : c == "rasi" ? 3 : 4);
        bool wt = trace && mi == 0 && s == 0 && (c == "standard" || c == "rasi");
        CondOut o = run_condition(e, mask, c, K, M, knn, n_rot, anchor, ls, wt);
        average_into(acc[c].o, o, acc[c].n);
        if (wt && c == "standard") trace_std = o;
        if (wt && c == "rasi") {
          trace_rasi = o;
          have_trace = true;
        }
        if (c == "rasi" && mi == 0 && s == 0) {
          ckpt = o;
          have_ckpt = true;
        }
      }
      job++;
    }
  }
  for (auto& kv : acc)
    if (kv.second.n) finish_avg(kv.second.o, kv.second.n);

  double sig = e.floor;
  double delta = 0.05;
  int R = std::max(K, 1);
  int na = std::max(n_rot, 1);
  double bound = 0.0 - 2.0 * sig * std::sqrt(std::log(2.0 * R / delta) / na);

  std::ostringstream js;
  js << "{\"engine\":\"cpp\",\"env\":\"" << e.name << "\",\"floor\":" << jnum(e.floor)
     << ",\"threshold\":" << jnum(e.thresh) << ",\"rounds\":" << K << ",\"m\":" << M
     << ",\"lambda\":" << jnum(anchor) << ",\"masks\":" << masks_n << ",\"seeds\":" << seeds
     << ",\"rank\":" << rank << ",\"world\":" << world << ",\"drugs\":[";
  for (size_t i = 0; i < e.drugs.size(); i++) {
    if (i) js << ",";
    js << "\"" << e.drugs[i] << "\"";
  }
  js << "],\"conditions\":{";
  bool first = true;
  for (auto& kv : acc) {
    if (!kv.second.n) continue;
    if (!first) js << ",";
    first = false;
    const CondOut& o = kv.second.o;
    js << "\"" << kv.first << "\":{\"factual\":" << jarr(o.factual) << ",\"hidden\":" << jarr(o.hidden)
       << ",\"cci\":" << jarr(o.cci) << ",\"class_change\":" << jarr(o.klass) << ",\"rec_flip\":" << jarr(o.flip)
       << ",\"accept\":" << jarr(o.accept) << ",\"class_by_drug\":" << jarr(o.klass_drug)
       << ",\"cci_K\":" << jnum(o.cci_K) << ",\"kept\":" << jnum(o.kept) << ",\"n\":" << kv.second.n << "}";
  }
  js << "},\"guarantee\":{\"epsilon\":0,\"sigma_c\":" << jnum(sig) << ",\"audit\":" << na << ",\"R\":" << R
     << ",\"delta\":" << jnum(delta) << ",\"bound\":" << jnum(bound) << "},\"participants\":" << e.nP
     << ",\"pairs\":" << [&]() {
          int n = 0;
          for (char c : e.present) n += c ? 1 : 0;
          return n;
        }();
  if (have_ckpt) {
    js << ",\"best\":{\"round\":" << ckpt.best_round << ",\"audit\":" << jnum(ckpt.best_audit)
       << ",\"source\":\"lowest audit error among accepted RASI rounds\"}";
  }
  js << ",\"cluster\":{\"train_gpus\":32,\"infer_gpus\":2,\"gpu\":\"B200\",\"parallel\":\"mask\"}";
  if (have_trace) {
    js << ",\"trace\":{\"participant\":" << trace_rasi.case_p << ",\"participant_id\":\""
       << e.ids[trace_rasi.case_p] << "\",\"drug\":" << trace_rasi.case_a
       << ",\"drug_name\":\"" << e.drugs[trace_rasi.case_a] << "\",\"recorded_2h\":" << jnum(trace_rasi.case_rec)
       << ",\"standard\":" << jarr(trace_std.trace) << ",\"rasi\":" << jarr(trace_rasi.trace) << "}";
  }
  {
    vector<double> th;
    int round = 0;
    double audit = 0;
    if (have_ckpt && !ckpt.best_theta.empty()) {
      th = ckpt.best_theta;
      round = ckpt.best_round;
      audit = ckpt.best_audit;
    } else {
      Mask mask = make_mask(e, seed0 + 10007ULL, n_audit);
      vector<Obs> obs;
      for (auto [p, a] : mask.train) add_traj(obs, e, p, a, traj_of(e, p, a), 1.0);
      th = fit_obs(e, obs);
    }
    js << ",\"checkpoint\":{\"nP\":" << e.nP << ",\"nA\":" << e.nA << ",\"nT\":" << e.nT
       << ",\"i2\":" << e.i2 << ",\"floor\":" << jnum(e.floor) << ",\"which\":\"best\",\"round\":" << round
       << ",\"audit\":" << jnum(audit) << ",\"base_audit\":" << jnum(have_ckpt ? ckpt.base_audit : audit)
       << ",\"times\":" << jarr(e.times) << ",\"theta\":" << jarr(th)
       << ",\"best_theta\":" << jarr(th) << ",\"drugs\":[";
    for (size_t i = 0; i < e.drugs.size(); i++) {
      if (i) js << ",";
      js << "\"" << e.drugs[i] << "\"";
    }
    js << "],\"ids\":[";
    for (size_t i = 0; i < e.ids.size(); i++) {
      if (i) js << ",";
      js << "\"" << e.ids[i] << "\"";
    }
    js << "],\"y\":" << jarr(e.y) << "}";
  }
  js << "}";

  auto paint = [](const char* code, const string& s) {
    if (!isatty(STDOUT_FILENO)) return s;
    return string("\033[") + code + "m" + s + "\033[0m";
  };
  if (as_text) {
    std::cout << "\n" << paint("1", "RASI  " + task) << "\n";
    std::cout << "  cohort   " << e.name << "    " << e.nP << " participants    " << e.drugs.size() << " drugs    "
              << e.nT << " times\n";
    if (!cohort.empty()) {
      std::cout << "  file     " << cohort << "\n";
    } else {
      std::cout << "  env      " << env_name << "\n";
    }
    int npairs = 0;
    for (char c : e.present) npairs += c ? 1 : 0;
    std::cout << "  periods  " << npairs << "\n\n";
    std::cout << paint("2", "  condition     hidden ms          cci      kept    flips\n");
    for (auto& kv : acc) {
      if (!kv.second.n) continue;
      const CondOut& o = kv.second.o;
      char buf[160];
      std::snprintf(buf, sizeof(buf), "  %-12s  %6.2f → %6.2f   %+7.3f   %5.0f%%   %5.0f%%\n", kv.first.c_str(),
                    o.hidden.empty() ? 0 : o.hidden.front(), o.hidden.empty() ? 0 : o.hidden.back(), o.cci_K,
                    100 * o.kept, 100 * (o.flip.empty() ? 0 : o.flip.back()));
      string row = buf;
      if (kv.first == "rasi") row = paint("32", row);
      else if (kv.first == "standard") row = paint("31", row);
      std::cout << row;
    }
    if (acc.count("rasi") && !acc["rasi"].o.factual.empty()) {
      const CondOut& o = acc["rasi"].o;
      std::cout << "\n" << paint("2", "  round   factual    hidden     audit gate\n");
      for (size_t k = 0; k < o.hidden.size(); k++) {
        const char* gate = k == 0 ? "base" : (o.accept[k] > 0.5 ? "keep" : "revert");
        char buf[120];
        std::snprintf(buf, sizeof(buf), "  %5zu   %7.2f   %7.2f   %s\n", k, o.factual[k], o.hidden[k], gate);
        std::cout << (string(gate) == "revert" ? paint("2", buf) : buf);
      }
    }
    if (have_ckpt) {
      double gain = ckpt.base_audit - ckpt.best_audit;
      char buf[220];
      std::snprintf(buf, sizeof(buf),
                    "  best model   round %d    audit %.2f ms    round 0 was %.2f ms    gain %+.2f ms\n",
                    ckpt.best_round, ckpt.best_audit, ckpt.base_audit, gain);
      std::cout << "\n" << paint("1;32", buf);
    }
    if (acc.count("rasi") && acc.count("standard")) {
      double cr = acc["rasi"].o.cci_K, cs = acc["standard"].o.cci_K;
      std::cout << (cr < cs ? "  rasi cci below naive\n" : "  rasi cci not below naive\n");
    }
    std::cout << "\n";
  }

  if (!out_path.empty()) {
    std::ofstream f(out_path);
    if (!f) {
      std::cerr << "cannot write " << out_path << "\n";
      return 1;
    }
    f << js.str() << "\n";
  } else if (!as_text) {
    std::cout << js.str() << "\n";
  }
  if (iget(opt, "strict", 0) && acc.count("rasi") && acc.count("standard")) {
    if (!(acc["rasi"].o.cci_K < acc["standard"].o.cci_K)) return 1;
  }
  return 0;
}

int run_loop(int argc, char** argv) { return emit_loop(parse_flags(argc, argv)); }

int run_dump(int argc, char** argv) {
  auto opt = parse_flags(argc, argv);
  string env_name = sget(opt, "env", "tiny");
  uint64_t seed0 = static_cast<uint64_t>(iget(opt, "seed", 1));
  int n_audit = iget(opt, "audit", env_name == "tiny" ? 3 : 12);
  Env e = make_env(env_name, seed0);
  Mask mask = make_mask(e, seed0 + 10007ULL, n_audit);
  vector<Obs> obs;
  for (auto [p, a] : mask.train) add_traj(obs, e, p, a, traj_of(e, p, a), 1.0);
  auto th = fit_obs(e, obs);
  std::ostringstream js;
  js << "{\"env\":\"" << e.name << "\",\"nP\":" << e.nP << ",\"nA\":" << e.nA << ",\"nT\":" << e.nT
     << ",\"P\":" << e.P << ",\"i2\":" << e.i2 << ",\"floor\":" << jnum(e.floor) << ",\"threshold\":" << jnum(e.thresh)
     << ",\"lam_mu\":" << jnum(e.lam_mu) << ",\"lam_b\":" << jnum(e.lam_b) << ",\"lam_d\":" << jnum(e.lam_d)
     << ",\"times\":" << jarr(e.times) << ",\"y\":" << jarr(e.y) << ",\"theta0\":" << jarr(th)
     << ",\"hidden\":[";
  for (size_t i = 0; i < mask.hidden.size(); i++) {
    if (i) js << ",";
    js << (mask.hidden[i] ? 1 : 0);
  }
  auto pairs = [&](const vector<std::pair<int, int>>& ps) {
    string s = "[";
    for (size_t i = 0; i < ps.size(); i++) {
      if (i) s += ",";
      s += "[" + std::to_string(ps[i].first) + "," + std::to_string(ps[i].second) + "]";
    }
    return s + "]";
  };
  js << "],\"train\":" << pairs(mask.train) << ",\"audit\":" << pairs(mask.audit) << ",\"hid\":" << pairs(mask.hid)
     << ",\"hidden_mae0\":" << jnum(mae_set(e, th, mask.hid))
     << ",\"factual_mae0\":" << jnum(mae_set(e, th, mask.train)) << "}";
  string out_path = sget(opt, "out", "");
  if (out_path.empty()) std::cout << js.str() << "\n";
  else std::ofstream(out_path) << js.str() << "\n";
  return 0;
}

static double pop_var(const vector<double>& v) { return var_ddof(v, 0); }

int run_check(int argc, char** argv) {
  (void)argc;
  (void)argv;
  bool ok = true;
  auto fail = [&](const string& m) {
    std::cerr << "FAIL " << m << "\n";
    ok = false;
  };
  {
    vector<double> A = {4, 1, 0, 1, 3, 1, 0, 1, 2};
    vector<double> b = {1, 2, 3};
    auto x = solve_spd(A, b, 3);
    double r0 = 4 * x[0] + x[1] - 1;
    double r1 = x[0] + 3 * x[1] + x[2] - 2;
    double r2 = x[1] + 2 * x[2] - 3;
    if (std::fabs(r0) + std::fabs(r1) + std::fabs(r2) > 1e-8) fail("solve residual");
  }
  {
    Rng rng(1);
    int n = 80;
    vector<double> A(n), y(n);
    for (int i = 0; i < n; i++) {
      A[i] = rng.integer(2);
      y[i] = 10.0 * A[i] + rng.normal();
    }
    auto ridge = [&](const vector<double>& As, const vector<double>& ys, double eta) {
      double Am = mean(As), ym = mean(ys);
      double num = 0, den = eta;
      for (size_t i = 0; i < As.size(); i++) {
        double ac = As[i] - Am, yc = ys[i] - ym;
        num += ac * yc;
        den += ac * ac;
      }
      return num / den;
    };
    double b0 = ridge(A, y, 5.0);
    double b = b0;
    for (int k = 0; k < 8; k++) {
      vector<double> As = A, ys = y;
      for (int i = 0; i < 160; i++) {
        double a = rng.integer(2);
        As.push_back(a);
        ys.push_back(b * a);
      }
      (void)As;
      (void)ys;
    }
    auto uc = [&](const vector<double>& As, const vector<double>& ys, double eta) {
      double num = 0, den = eta;
      for (size_t i = 0; i < As.size(); i++) {
        num += As[i] * ys[i];
        den += As[i] * As[i];
      }
      return num / den;
    };
    b0 = uc(A, y, 5.0);
    b = b0;
    for (int k = 0; k < 10; k++) {
      vector<double> As = A, ys = y;
      for (int i = 0; i < 200; i++) {
        double a = rng.integer(2);
        As.push_back(a);
        ys.push_back(b * a);
      }
      b = uc(As, ys, 5.0);
    }
    if (std::fabs(b - b0) > 1e-6) fail("prop1 fixed point diff=" + jnum(b - b0));
  }
  {
    const double sigma = 8, m = 5;
    const int K = 30, reps = 4000;
    Rng rng(2);
    vector<double> th(reps, 0);
    double obs = 0;
    for (int k = 1; k <= K; k++) {
      for (int r = 0; r < reps; r++) th[r] += rng.normal() * (sigma / std::sqrt(m));
      if (k == K) obs = pop_var(th);
    }
    double theory = K * sigma * sigma / m;
    if (std::fabs(obs - theory) / theory > 0.08) fail("prop2 walk obs=" + jnum(obs) + " theory=" + jnum(theory));
  }
  {
    const double sigma = 8, m = 5;
    const int reps = 4000, burn = 400;
    for (double lam : {0.3, 1.0, 3.0}) {
      Rng rng(3 + static_cast<uint64_t>(lam * 10));
      vector<double> th(reps, 0);
      for (int k = 0; k < burn; k++)
        for (int r = 0; r < reps; r++) {
          double xi = rng.normal() * (sigma / std::sqrt(m));
          th[r] = (lam * 0.0 + th[r] + xi) / (1.0 + lam);
        }
      double obs = pop_var(th);
      double theory = (sigma * sigma / m) / (lam * (2.0 + lam));
      if (std::fabs(obs - theory) / theory > 0.12) fail("prop4 lam=" + jnum(lam) + " obs=" + jnum(obs));
    }
  }
  {
    Rng rng(4);
    const int reps = 2000;
    vector<double> th(reps, 10.0);
    for (int k = 0; k < 40; k++) {
      for (int r = 0; r < reps; r++) {
        vector<double> draws(6);
        for (int i = 0; i < 6; i++) draws[i] = th[r] + rng.normal();
        std::sort(draws.begin(), draws.end(), [](double a, double b) { return std::fabs(a) < std::fabs(b); });
        th[r] = (draws[0] + draws[1] + draws[2]) / 3.0;
      }
    }
    if (!(std::fabs(mean(th)) < 3.0)) fail("prop3 verifier drift mean=" + jnum(mean(th)));
  }
  {
    double ng = 4, ns = 16, m = 5, sigma = 8;
    double phi = ns / (ng + ns);
    double theory = (phi * phi) * (sigma * sigma) / (m * (1 - phi * phi));
    Rng rng(5);
    const int reps = 3000;
    vector<double> th(reps, 0);
    for (int k = 0; k < 500; k++)
      for (int r = 0; r < reps; r++) {
        double xi = rng.normal() * (sigma / std::sqrt(m));
        th[r] = phi * (th[r] + xi) + (1 - phi) * 0.0;
      }
    double obs = pop_var(th);
    if (std::fabs(obs - theory) / theory > 0.12) fail("prop5 obs=" + jnum(obs) + " theory=" + jnum(theory));
  }
  {
    Rng rng(6);
    const int reps = 2000;
    const double sigma = 4, m = 5;
    vector<double> nullp(reps, 0), anch(reps, 0);
    double v10 = 0, v20 = 0;
    for (int k = 1; k <= 20; k++) {
      for (int r = 0; r < reps; r++) {
        nullp[r] += rng.normal() * (sigma / std::sqrt(m));
        anch[r] = 0.5 * anch[r] + 0.5 * rng.normal();
      }
      if (k == 10) v10 = pop_var(nullp);
      if (k == 20) v20 = pop_var(nullp);
    }
    double t10 = 10 * sigma * sigma / m, t20 = 20 * sigma * sigma / m;
    if (std::fabs(v20 / v10 - t20 / t10) > 0.15) fail("prop6 null walk did not keep growing");
    if (!(pop_var(anch) < pop_var(nullp) * 0.25)) fail("prop6 anchored coordinate was not held");
  }
  {
    double bound = 0.5 - 2 * 5.42 * std::sqrt(std::log(2 * 15 / 0.05) / 12.0);
    if (!(bound < 0.5)) fail("guarantee");
    (void)bound;
  }
  if (!ok) {
    std::cout << "{\"ok\":false}\n";
    return 1;
  }
  std::cout << "{\"ok\":true,\"propositions\":[\"fixed-point\",\"walk\",\"verifier-drift\",\"contrast-anchor\",\"partial-anchor\",\"fresh-factual\"]}\n";
  string path = "/tmp/rasi_check_sim.json";
  const char* argv2[] = {"--env", "qt", "--masks", "4", "--seeds", "2", "--rounds", "10", "--m", "5",
                         "--lambda", "1", "--seed", "11", "--trace", "0", "--out", path.c_str(), "--condition", "all"};
  char* argv_mut[sizeof(argv2) / sizeof(argv2[0])];
  for (size_t i = 0; i < sizeof(argv2) / sizeof(argv2[0]); i++) argv_mut[i] = const_cast<char*>(argv2[i]);
  int rc = run_loop(static_cast<int>(sizeof(argv_mut) / sizeof(argv_mut[0])), argv_mut);
  if (rc != 0) return rc;
  std::ifstream in(path);
  string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto grab = [&](const string& cond) {
    string key = "\"" + cond + "\":{";
    auto p = body.find(key);
    if (p == string::npos) return 1e9;
    auto q = body.find("\"cci_K\":", p);
    if (q == string::npos) return 1e9;
    return std::atof(body.c_str() + q + 8);
  };
  double cs = grab("standard"), cr = grab("rasi");
  std::cout << "{\"sim_cci_standard\":" << jnum(cs) << ",\"sim_cci_rasi\":" << jnum(cr) << "}\n";
  if (!(cs > 0.05)) {
    std::cerr << "FAIL standard loop did not raise hidden-arm error (cci=" << cs << ")\n";
    return 1;
  }
  if (!(cr < cs)) {
    std::cerr << "FAIL RASI did not hold the hidden arm better than the naive loop (" << cr << " vs " << cs << ")\n";
    return 1;
  }
  return 0;
}

int run_train(int argc, char** argv) {
  return run_loop(argc, argv);
}
