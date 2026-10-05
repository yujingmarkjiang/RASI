#pragma once
#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>

struct Rng {
  uint64_t s[4];
  bool spare_ok = false;
  double spare = 0;

  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  explicit Rng(uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) s[i] = splitmix(x);
  }
  uint64_t next() {
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }
  int integer(int n) {
    if (n <= 1) return 0;
    uint64_t bound = static_cast<uint64_t>(n);
    uint64_t limit = (UINT64_MAX / bound) * bound;
    for (;;) {
      uint64_t v = next();
      if (v < limit) return static_cast<int>(v % bound);
    }
  }
  double uniform() { return (next() >> 11) * 0x1.0p-53; }
  double normal() {
    if (spare_ok) {
      spare_ok = false;
      return spare;
    }
    double u1 = uniform();
    if (u1 < 1e-12) u1 = 1e-12;
    double u2 = uniform();
    double r = std::sqrt(-2.0 * std::log(u1));
    constexpr double pi2 = 6.28318530717958647692;
    spare = r * std::sin(pi2 * u2);
    spare_ok = true;
    return r * std::cos(pi2 * u2);
  }
  template <class T>
  void shuffle(std::vector<T>& a) {
    for (int i = static_cast<int>(a.size()) - 1; i > 0; i--) {
      int j = integer(i + 1);
      std::swap(a[i], a[j]);
    }
  }
};
