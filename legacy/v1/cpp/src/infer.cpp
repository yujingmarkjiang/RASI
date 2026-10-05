#include "engine.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using std::string;
using std::vector;

static std::vector<double> array_after(const string& body, size_t from, const string& key) {
  auto p = body.find("\"" + key + "\"", from);
  if (p == string::npos) return {};
  auto lb = body.find('[', p);
  if (lb == string::npos) return {};
  auto rb = body.find(']', lb);
  if (rb == string::npos) return {};
  vector<double> v;
  string cur;
  for (size_t i = lb + 1; i < rb; i++) {
    char c = body[i];
    if (c == ',' || c == ' ' || c == '\n') {
      if (!cur.empty()) {
        v.push_back(std::atof(cur.c_str()));
        cur.clear();
      }
    } else cur.push_back(c);
  }
  if (!cur.empty()) v.push_back(std::atof(cur.c_str()));
  return v;
}
static int int_after(const string& body, size_t from, const string& key, int dflt) {
  auto p = body.find("\"" + key + "\"", from);
  if (p == string::npos) return dflt;
  auto c = body.find(':', p);
  if (c == string::npos) return dflt;
  return std::atoi(body.c_str() + c + 1);
}

static double dbl_after(const string& body, size_t from, const string& key, double dflt) {
  auto p = body.find("\"" + key + "\"", from);
  if (p == string::npos) return dflt;
  auto c = body.find(':', p);
  if (c == string::npos) return dflt;
  return std::atof(body.c_str() + c + 1);
}

static void emit(int device, const string& route, const string& stage, const string& field, double y, int round,
                 double audit, double base) {
  if (isatty(STDOUT_FILENO)) {
    std::cout << "\n\033[1mRASI  infer\033[0m\n";
    std::cout << "  device      " << device << "    route " << route << "    stage " << stage << "\n";
    std::cout << "  checkpoint  best accepted round " << round << "\n";
    if (base > 0 || audit > 0) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "  audit       %.2f ms    round 0 was %.2f ms    gain %+.2f ms\n", audit, base,
                    base - audit);
      std::cout << "\033[32m" << buf << "\033[0m";
    }
    std::cout << "\033[1;32m  " << field << "   " << y << " ms\033[0m\n\n";
    return;
  }
  std::cout << "{\"device\":" << device << ",\"route\":\"" << route << "\",\"stage\":\"" << stage << "\",\"" << field
            << "\":" << y << ",\"round\":" << round << ",\"audit\":" << audit << ",\"base_audit\":" << base << "}\n";
}

int run_infer(int argc, char** argv) {
  string ckpt, stage = "full", route = "shard";
  int device = 0, devices = 2, p = 0, a = 0, t = 0;
  double part = 0;
  bool have_part = false;
  for (int i = 0; i < argc; i++) {
    string k = argv[i];
    if (k == "--ckpt") ckpt = argv[++i];
    else if (k == "--stage") stage = argv[++i];
    else if (k == "--route") route = argv[++i];
    else if (k == "--device") device = std::atoi(argv[++i]);
    else if (k == "--devices") devices = std::atoi(argv[++i]);
    else if (k == "--p") p = std::atoi(argv[++i]);
    else if (k == "--a") a = std::atoi(argv[++i]);
    else if (k == "--t") t = std::atoi(argv[++i]);
    else if (k == "--part") {
      part = std::atof(argv[++i]);
      have_part = true;
    }
  }
  if (devices < 1) devices = 1;
  std::ifstream in(ckpt);
  if (!in) {
    std::cerr << "missing checkpoint\n";
    return 1;
  }
  string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  size_t from = body.find("\"checkpoint\"");
  if (from == string::npos) from = 0;
  int nP = int_after(body, from, "nP", 0);
  int nA = int_after(body, from, "nA", 0);
  int nT = int_after(body, from, "nT", 0);
  auto theta = array_after(body, from, "best_theta");
  if (theta.empty()) theta = array_after(body, from, "theta");
  int best_round = int_after(body, from, "round", 0);
  double audit = dbl_after(body, from, "audit", 0);
  double base_audit = dbl_after(body, from, "base_audit", audit);
  if (nP <= 0 || nA <= 0 || nT <= 0 || theta.empty()) {
    std::cerr << "bad checkpoint\n";
    return 1;
  }
  if (p < 0 || p >= nP || a < 0 || a >= nA || t < 0 || t >= nT) {
    std::cerr << "request out of range\n";
    return 2;
  }
  auto mu = [&](int aa, int tt) { return aa * nT + tt; };
  auto beta = [&](int pp) { return nA * nT + pp; };
  auto delta = [&](int pp, int aa) { return nA * nT + nP + (pp * nA + aa); };
  if (route == "replica") {
    if ((p % devices) != device) {
      std::cerr << "wrong shard\n";
      return 3;
    }
    double y = theta[mu(a, t)] + theta[beta(p)] + theta[delta(p, a)];
    emit(device, "replica", "full", "prediction", y, best_round, audit, base_audit);
    return 0;
  }
  if (stage == "encode") {
    if (device != 0) {
      std::cerr << "encode is device 0\n";
      return 3;
    }
    emit(0, "shard", "encode", "part", theta[mu(a, t)], best_round, audit, base_audit);
    return 0;
  }
  if (stage == "rollout") {
    if (devices > 1 && device != 1) {
      std::cerr << "rollout is device 1\n";
      return 3;
    }
    if (!have_part) {
      std::cerr << "rollout needs --part from device 0\n";
      return 2;
    }
    double y = part + theta[beta(p)] + theta[delta(p, a)];
    emit(device, "shard", "rollout", "prediction", y, best_round, audit, base_audit);
    return 0;
  }
  double y = theta[mu(a, t)] + theta[beta(p)] + theta[delta(p, a)];
  emit(device, "shard", "full", "prediction", y, best_round, audit, base_audit);
  return 0;
}
