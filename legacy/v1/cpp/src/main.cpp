#include "engine.hpp"
#include "rng.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

static void usage() {
  std::cerr <<
      "RASI\n"
      "\n"
      "  train    fit on a cohort csv\n"
      "  test     naive self-training against rasi\n"
      "  infer    score a checkpoint\n"
      "  apply    keep the best audit round\n"
      "  check    fixed point, walk, anchor\n"
      "\n"
      "  rasi train --cohort data/e1_qt_cohort.csv\n"
      "  rasi test  --cohort data/e1_qt_cohort.csv --masks 2 --rounds 6\n"
      "  rasi infer --ckpt run.json --p 0 --a 2 --t 3\n"
      "\n"
      "  python3 python/modalities.py list\n"
      "  python3 python/modalities.py apply --cohort data/e1_qt_cohort.csv\n";
}

static bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 0; i < argc; i++)
    if (std::string(argv[i]) == name) return true;
  return false;
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "check";
  if (cmd == "help" || cmd == "-h" || cmd == "--help") {
    usage();
    return 0;
  }
  if (cmd == "rng") {
    uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
    Rng r(seed);
    for (int i = 0; i < 4; i++) std::printf("%llu\n", static_cast<unsigned long long>(r.next()));
    for (int i = 0; i < 4; i++) std::printf("%.17g\n", r.normal());
    return 0;
  }
  std::vector<std::string> store;
  std::vector<char*> owned;
  int n = 0;
  char** rest = argv + 2;
  if (argc > 2) n = argc - 2;
  const char* cohort = std::getenv("RASI_COHORT");
  std::string cohort_store;
  if (!cohort || !*cohort) {
    const char* cands[] = {"data/e1_qt_cohort.csv", "rasi/data/e1_qt_cohort.csv"};
    cohort = cands[1];
    for (const char* c : cands) {
      if (access(c, R_OK) == 0) {
        cohort = c;
        break;
      }
    }
    char exe[4096];
    ssize_t nread = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (nread > 0) {
      exe[nread] = 0;
      std::string dir(exe);
      auto slash = dir.find_last_of('/');
      if (slash != std::string::npos) dir.resize(slash);
      std::string beside = dir + "/../../data/e1_qt_cohort.csv";
      if (access(beside.c_str(), R_OK) == 0) {
        cohort_store = beside;
        cohort = cohort_store.c_str();
      }
    }
  }
  bool inject = (cmd == "train" || cmd == "test" || cmd == "apply") && !has_flag(n, rest, "--cohort") &&
                !has_flag(n, rest, "--env");
  if (inject) {
    for (int i = 0; i < n; i++) store.push_back(rest[i]);
    store.push_back("--cohort");
    store.push_back(cohort);
    store.push_back("--task");
    store.push_back(cmd == "apply" ? "apply" : cmd);
    owned.reserve(store.size());
    for (auto& s : store) owned.push_back(s.data());
    rest = owned.data();
    n = static_cast<int>(owned.size());
  }
  if (cmd == "loop" || cmd == "train" || cmd == "test") return run_loop(n, rest);
  if (cmd == "apply") {
    std::string ckpt = "/tmp/rasi_apply.json";
    std::vector<std::string> train;
    for (int i = 0; i < n; i++) train.push_back(rest[i]);
    train.push_back("--format");
    train.push_back(isatty(STDOUT_FILENO) ? "text" : "json");
    train.push_back("--out");
    train.push_back(ckpt);
    train.push_back("--trace");
    train.push_back("0");
    if (!has_flag(n, rest, "--masks")) {
      train.push_back("--masks");
      train.push_back("2");
    }
    if (!has_flag(n, rest, "--rounds")) {
      train.push_back("--rounds");
      train.push_back("6");
    }
    if (!has_flag(n, rest, "--seeds")) {
      train.push_back("--seeds");
      train.push_back("1");
    }
    std::vector<char*> av;
    for (auto& s : train) av.push_back(s.data());
    int rc = run_loop(static_cast<int>(av.size()), av.data());
    if (rc != 0) return rc;
    std::vector<std::string> inf = {"--ckpt", ckpt, "--stage", "full", "--device", "0", "--devices", "2"};
    auto take = [&](const char* name, const char* dflt) {
      for (int i = 0; i < n; i++)
        if (std::string(rest[i]) == name && i + 1 < n) return std::string(rest[i + 1]);
      return std::string(dflt);
    };
    inf.push_back("--p");
    inf.push_back(take("--p", "0"));
    inf.push_back("--a");
    inf.push_back(take("--a", "2"));
    inf.push_back("--t");
    inf.push_back(take("--t", "3"));
    std::vector<char*> iv;
    for (auto& s : inf) iv.push_back(s.data());
    return run_infer(static_cast<int>(iv.size()), iv.data());
  }
  if (cmd == "check") return run_check(n, rest);
  if (cmd == "dump") return run_dump(n, rest);
  if (cmd == "gru") return run_gru(n, rest);
  if (cmd == "infer") return run_infer(n, rest);
  usage();
  return 2;
}