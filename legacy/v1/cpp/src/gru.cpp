#include "engine.hpp"
#include "rng.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

struct GRU {
  int I, H, T;
  vector<double> Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wo, bo;
  explicit GRU(int i, int h, uint64_t seed) : I(i), H(h), T(0) {
    Rng rng(seed);
    auto init = [&](int n, double s) {
      vector<double> v(n);
      for (int k = 0; k < n; k++) v[k] = rng.normal() * s;
      return v;
    };
    double sI = 0.1, sH = 0.1;
    Wz = init(H * I, sI);
    Wr = init(H * I, sI);
    Wh = init(H * I, sI);
    Uz = init(H * H, sH);
    Ur = init(H * H, sH);
    Uh = init(H * H, sH);
    bz = init(H, 0.0);
    br = init(H, 0.0);
    bh = init(H, 0.0);
    Wo = init(H, 0.1);
    bo = {0.0};
  }
  vector<double>* param(int which, int& n) {
    vector<double>* ps[] = {&Wz, &Wr, &Wh, &Uz, &Ur, &Uh, &bz, &br, &bh, &Wo, &bo};
    n = static_cast<int>(ps[which]->size());
    return ps[which];
  }
  int nparam() const {
    return 3 * H * I + 3 * H * H + 3 * H + H + 1;
  }
};

static double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Cache {
  vector<double> hprev, z, r, hh, hnew, rh;
};

static double forward(const GRU& g, const vector<double>& X, const vector<double>& y, vector<Cache>* cache) {
  int I = g.I, H = g.H, T = static_cast<int>(y.size());
  vector<double> h(H, 0.0);
  double loss = 0;
  if (cache) cache->assign(T, {});
  for (int t = 0; t < T; t++) {
    const double* x = X.data() + t * I;
    Cache c;
    c.hprev = h;
    c.z.resize(H);
    c.r.resize(H);
    c.hh.resize(H);
    c.hnew.resize(H);
    c.rh.resize(H);
    for (int i = 0; i < H; i++) {
      double az = g.bz[i], ar = g.br[i], ah = g.bh[i];
      for (int j = 0; j < I; j++) {
        az += g.Wz[i * I + j] * x[j];
        ar += g.Wr[i * I + j] * x[j];
        ah += g.Wh[i * I + j] * x[j];
      }
      for (int j = 0; j < H; j++) {
        az += g.Uz[i * H + j] * h[j];
        ar += g.Ur[i * H + j] * h[j];
      }
      c.z[i] = sig(az);
      c.r[i] = sig(ar);
      c.rh[i] = c.r[i] * h[i];
    }
    for (int i = 0; i < H; i++) {
      double ah = g.bh[i];
      for (int j = 0; j < I; j++) ah += g.Wh[i * I + j] * x[j];
      for (int j = 0; j < H; j++) ah += g.Uh[i * H + j] * c.rh[j];
      c.hh[i] = std::tanh(ah);
      c.hnew[i] = (1 - c.z[i]) * h[i] + c.z[i] * c.hh[i];
    }
    double o = g.bo[0];
    for (int i = 0; i < H; i++) o += g.Wo[i] * c.hnew[i];
    double d = o - y[t];
    loss += 0.5 * d * d;
    h = c.hnew;
    if (cache) (*cache)[t] = c;
  }
  return loss;
}

struct Grads {
  vector<double> Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wo, bo;
  explicit Grads(const GRU& g)
      : Wz(g.Wz.size()), Wr(g.Wr.size()), Wh(g.Wh.size()), Uz(g.Uz.size()), Ur(g.Ur.size()), Uh(g.Uh.size()),
        bz(g.bz.size()), br(g.br.size()), bh(g.bh.size()), Wo(g.Wo.size()), bo(1) {}
  void zero() {
    auto z = [](vector<double>& v) { std::fill(v.begin(), v.end(), 0.0); };
    z(Wz); z(Wr); z(Wh); z(Uz); z(Ur); z(Uh); z(bz); z(br); z(bh); z(Wo); z(bo);
  }
  void add(const Grads& o) {
    auto a = [](vector<double>& d, const vector<double>& s) { for (size_t i = 0; i < d.size(); i++) d[i] += s[i]; };
    a(Wz, o.Wz); a(Wr, o.Wr); a(Wh, o.Wh); a(Uz, o.Uz); a(Ur, o.Ur); a(Uh, o.Uh);
    a(bz, o.bz); a(br, o.br); a(bh, o.bh); a(Wo, o.Wo); a(bo, o.bo);
  }
  void scale(double s) {
    auto f = [&](vector<double>& v) { for (double& x : v) x *= s; };
    f(Wz); f(Wr); f(Wh); f(Uz); f(Ur); f(Uh); f(bz); f(br); f(bh); f(Wo); f(bo);
  }
  vector<double> flat() const {
    vector<double> o;
    auto app = [&](const vector<double>& v) { o.insert(o.end(), v.begin(), v.end()); };
    app(Wz); app(Wr); app(Wh); app(Uz); app(Ur); app(Uh); app(bz); app(br); app(bh); app(Wo); app(bo);
    return o;
  }
};

static Grads backward(const GRU& g, const vector<double>& X, const vector<double>& y) {
  vector<Cache> cache;
  forward(g, X, y, &cache);
  int I = g.I, H = g.H, T = static_cast<int>(y.size());
  Grads grad(g);
  vector<double> dh_next(H, 0.0);
  for (int t = T - 1; t >= 0; t--) {
    const double* x = X.data() + t * I;
    const Cache& c = cache[t];
    double o = g.bo[0];
    for (int i = 0; i < H; i++) o += g.Wo[i] * c.hnew[i];
    double dLdo = o - y[t];
    grad.bo[0] += dLdo;
    vector<double> dh = dh_next;
    for (int i = 0; i < H; i++) {
      grad.Wo[i] += dLdo * c.hnew[i];
      dh[i] += dLdo * g.Wo[i];
    }
    vector<double> dz(H), dhh(H), dhprev(H, 0.0), drh(H, 0.0);
    for (int i = 0; i < H; i++) {
      dz[i] = dh[i] * (c.hh[i] - c.hprev[i]);
      dhh[i] = dh[i] * c.z[i];
      dhprev[i] = dh[i] * (1.0 - c.z[i]);
    }
    vector<double> dah(H);
    for (int i = 0; i < H; i++) dah[i] = dhh[i] * (1.0 - c.hh[i] * c.hh[i]);
    for (int i = 0; i < H; i++) {
      grad.bh[i] += dah[i];
      for (int j = 0; j < I; j++) grad.Wh[i * I + j] += dah[i] * x[j];
      for (int j = 0; j < H; j++) grad.Uh[i * H + j] += dah[i] * c.rh[j];
    }
    for (int j = 0; j < H; j++) {
      double s = 0;
      for (int i = 0; i < H; i++) s += g.Uh[i * H + j] * dah[i];
      drh[j] += s;
    }
    vector<double> dr(H);
    for (int i = 0; i < H; i++) {
      dr[i] = drh[i] * c.hprev[i];
      dhprev[i] += drh[i] * c.r[i];
    }
    auto sigp = [](double z) { return z * (1.0 - z); };
    vector<double> dar(H), daz(H);
    for (int i = 0; i < H; i++) {
      dar[i] = dr[i] * sigp(c.r[i]);
      daz[i] = dz[i] * sigp(c.z[i]);
    }
    for (int i = 0; i < H; i++) {
      grad.br[i] += dar[i];
      grad.bz[i] += daz[i];
      for (int j = 0; j < I; j++) {
        grad.Wr[i * I + j] += dar[i] * x[j];
        grad.Wz[i * I + j] += daz[i] * x[j];
      }
      for (int j = 0; j < H; j++) {
        grad.Ur[i * H + j] += dar[i] * c.hprev[j];
        grad.Uz[i * H + j] += daz[i] * c.hprev[j];
      }
    }
    for (int j = 0; j < H; j++) {
      double sr = 0, sz = 0;
      for (int i = 0; i < H; i++) {
        sr += g.Ur[i * H + j] * dar[i];
        sz += g.Uz[i * H + j] * daz[i];
      }
      dhprev[j] += sr + sz;
    }
    dh_next = dhprev;
  }
  return grad;
}

static vector<double> toy_batch(int n, int I, int T, uint64_t seed, vector<double>& Y) {
  Rng rng(seed);
  vector<double> X(n * T * I);
  Y.assign(n * T, 0);
  for (int s = 0; s < n; s++)
    for (int t = 0; t < T; t++) {
      double sum = 0;
      for (int j = 0; j < I; j++) {
        double v = rng.normal();
        X[(s * T + t) * I + j] = v;
        sum += v;
      }
      Y[s * T + t] = 0.3 * sum + 0.1 * rng.normal();
    }
  return X;
}

static bool fd_check() {
  GRU g(3, 4, 9);
  vector<double> Y;
  auto X = toy_batch(1, g.I, 3, 21, Y);
  auto grad = backward(g, X, Y).flat();
  vector<int> idxs;
  int np = g.nparam();
  for (int k = 0; k < 6; k++) idxs.push_back((k * 17) % np);
  double eps = 1e-5;
  vector<double*> ptrs;
  vector<int> sizes;
  auto take = [&](vector<double>& v) {
    for (double& x : v) ptrs.push_back(&x);
  };
  take(g.Wz); take(g.Wr); take(g.Wh); take(g.Uz); take(g.Ur); take(g.Uh);
  take(g.bz); take(g.br); take(g.bh); take(g.Wo); take(g.bo);
  (void)sizes;
  double worst = 0;
  for (int idx : idxs) {
    double orig = *ptrs[idx];
    *ptrs[idx] = orig + eps;
    double lp = forward(g, X, Y, nullptr);
    *ptrs[idx] = orig - eps;
    double lm = forward(g, X, Y, nullptr);
    *ptrs[idx] = orig;
    double fd = (lp - lm) / (2 * eps);
    double err = std::fabs(fd - grad[idx]);
    double den = std::max(1.0, std::max(std::fabs(fd), std::fabs(grad[idx])));
    worst = std::max(worst, err / den);
  }
  if (worst > 1e-4) {
    std::cerr << "FAIL gru finite-difference rel " << worst << "\n";
    return false;
  }
  std::cout << "{\"gru_fd_rel\":" << worst << "}\n";
  return true;
}

static bool dp_identity() {
  GRU g(3, 4, 9);
  vector<double> Y;
  auto X = toy_batch(4, g.I, 3, 22, Y);
  int T = 3, I = g.I;
  Grads full(g);
  full.zero();
  for (int s = 0; s < 4; s++) {
    vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
    vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
    full.add(backward(g, Xs, Ys));
  }
  full.scale(1.0 / 4);
  Grads a(g), b(g);
  a.zero();
  b.zero();
  for (int s = 0; s < 2; s++) {
    vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
    vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
    a.add(backward(g, Xs, Ys));
  }
  a.scale(1.0 / 4);
  for (int s = 2; s < 4; s++) {
    vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
    vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
    b.add(backward(g, Xs, Ys));
  }
  b.scale(1.0 / 4);
  auto fa = full.flat(), aa = a.flat(), bb = b.flat();
  double worst = 0;
  for (size_t i = 0; i < fa.size(); i++) worst = std::max(worst, std::fabs(fa[i] - (aa[i] + bb[i])));
  if (worst > 1e-9) {
    std::cerr << "FAIL dp identity " << worst << "\n";
    return false;
  }
  double before = 0;
  for (int s = 0; s < 4; s++) {
    vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
    vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
    before += forward(g, Xs, Ys, nullptr);
  }
  vector<double*> ptrs;
  auto take = [&](vector<double>& v) { for (double& x : v) ptrs.push_back(&x); };
  take(g.Wz); take(g.Wr); take(g.Wh); take(g.Uz); take(g.Ur); take(g.Uh);
  take(g.bz); take(g.br); take(g.bh); take(g.Wo); take(g.bo);
  vector<double> m(ptrs.size()), v(ptrs.size());
  double lr = 0.05, b1 = 0.9, b2 = 0.999, wd = 0.0, eps = 1e-8;
  for (int step = 1; step <= 40; step++) {
    Grads gg(g);
    gg.zero();
    for (int s = 0; s < 4; s++) {
      vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
      vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
      gg.add(backward(g, Xs, Ys));
    }
    gg.scale(1.0 / 4);
    auto gf = gg.flat();
    for (size_t i = 0; i < ptrs.size(); i++) {
      m[i] = b1 * m[i] + (1 - b1) * gf[i];
      v[i] = b2 * v[i] + (1 - b2) * gf[i] * gf[i];
      double mh = m[i] / (1 - std::pow(b1, step));
      double vh = v[i] / (1 - std::pow(b2, step));
      *ptrs[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * (*ptrs[i]));
    }
  }
  double after = 0;
  for (int s = 0; s < 4; s++) {
    vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
    vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
    after += forward(g, Xs, Ys, nullptr);
  }
  std::cout << "{\"gru_loss_before\":" << before << ",\"gru_loss_after\":" << after << ",\"dp_abs\":" << worst << "}\n";
  if (!(after < before * 0.85)) {
    std::cerr << "FAIL gru loss did not fall\n";
    return false;
  }
  return true;
}

static void write_vec(const string& path, const vector<double>& v) {
  std::ofstream f(path, std::ios::binary);
  int n = static_cast<int>(v.size());
  f.write(reinterpret_cast<const char*>(&n), sizeof(n));
  f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(double));
}
static vector<double> read_vec(const string& path) {
  std::ifstream f(path, std::ios::binary);
  int n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  vector<double> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(double));
  return v;
}

static int file_allreduce(int rank, int world, const string& dir) {
  GRU g(3, 4, 9);
  vector<double> Y;
  auto X = toy_batch(4, g.I, 3, 22, Y);
  int T = 3, I = g.I;
  int n0 = (4 * rank) / world;
  int n1 = (4 * (rank + 1)) / world;
  Grads local(g);
  local.zero();
  for (int s = n0; s < n1; s++) {
    vector<double> Xs(X.begin() + s * T * I, X.begin() + (s + 1) * T * I);
    vector<double> Ys(Y.begin() + s * T, Y.begin() + (s + 1) * T);
    local.add(backward(g, Xs, Ys));
  }
  auto flat = local.flat();
  write_vec(dir + "/g_" + std::to_string(rank), flat);
  std::ofstream(dir + "/ready_" + std::to_string(rank)).close();
  for (int r = 0; r < world; r++) {
    string p = dir + "/ready_" + std::to_string(r);
    for (int spin = 0; spin < 4000; spin++) {
      std::ifstream in(p);
      if (in) break;
      if (spin == 3999) {
        std::cerr << "allreduce timeout\n";
        return 1;
      }
    }
  }
  vector<double> acc(flat.size(), 0);
  for (int r = 0; r < world; r++) {
    auto o = read_vec(dir + "/g_" + std::to_string(r));
    for (size_t i = 0; i < acc.size(); i++) acc[i] += o[i];
  }
  for (double& x : acc) x /= 4.0;
  write_vec(dir + "/out_" + std::to_string(rank), acc);
  return 0;
}

int run_gru(int argc, char** argv) {
  std::map<string, string> opt;
  for (int i = 0; i < argc; i++) {
    string a = argv[i];
    if (a.rfind("--", 0) != 0) continue;
    string key = a.substr(2);
    string val = "1";
    if (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) val = argv[++i];
    opt[key] = val;
  }
  if (opt.count("rank")) {
    int rank = std::atoi(opt["rank"].c_str());
    int world = std::atoi(opt["world-size"].c_str());
    return file_allreduce(rank, world, opt["dir"]);
  }
  bool ok = fd_check() && dp_identity();
  return ok ? 0 : 1;
}
