// The same normalised objective as python/rasi/models/linear.py.
// No synthetic toy environment, hidden-label access or implicit regularisation.
#include <cmath>
#include <vector>
#include <limits>
#include <cstddef>

extern "C" int rasi_solve(int n, int p, const double* X, const double* y, const double* weights,
                          int m, const double* D, const double* d, const double* cweights,
                          double lambda, double eta, double* result) noexcept {
  try {
    if (n <= 0 || p <= 0 || m < 0 || !X || !y || !weights || !result ||
        !std::isfinite(lambda) || !std::isfinite(eta) || lambda < 0 || eta < 0 ||
        (lambda > 0 && (m == 0 || !D || !d || !cweights))) return 1;
    const std::size_t P = static_cast<std::size_t>(p);
    if (P > std::numeric_limits<std::size_t>::max() / P) return 1;
    std::vector<double> G(P * P, 0.0), b(P, 0.0);
    auto add = [&](int rows, const double* x, const double* target, const double* ws, double mass) -> bool {
      if (mass == 0) return true;
      double total = 0;
      for (int i = 0; i < rows; ++i) {
        if (!std::isfinite(ws[i]) || ws[i] < 0 || !std::isfinite(target[i])) return false;
        total += ws[i];
      }
      if (!(total > 0) || !std::isfinite(total)) return false;
      for (int i = 0; i < rows; ++i) {
        const double* row = x + static_cast<std::size_t>(i) * P;
        const double w = mass * ws[i] / total;
        for (int j = 0; j < p; ++j) {
          if (!std::isfinite(row[j])) return false;
          b[j] += w * row[j] * target[i];
          for (int k = 0; k <= j; ++k) G[static_cast<std::size_t>(j) * P + k] += w * row[j] * row[k];
        }
      }
      return true;
    };
    if (!add(n, X, y, weights, 1.0) || !add(m, D, d, cweights, lambda)) return 2;
    for (int j = 0; j < p; ++j) G[static_cast<std::size_t>(j) * P + j] += eta;
    // In-place Cholesky; rank deficiency is an error, never silently patched.
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k <= j; ++k) {
        double value = G[static_cast<std::size_t>(j) * P + k];
        for (int h = 0; h < k; ++h) value -= G[static_cast<std::size_t>(j) * P + h] * G[static_cast<std::size_t>(k) * P + h];
        if (j == k) {
          if (!(value > 0) || !std::isfinite(value)) return 3;
          G[static_cast<std::size_t>(j) * P + k] = std::sqrt(value);
        } else G[static_cast<std::size_t>(j) * P + k] = value / G[static_cast<std::size_t>(k) * P + k];
      }
    }
    std::vector<double> z(P, 0.0);
    for (int j = 0; j < p; ++j) {
      double value = b[j];
      for (int k = 0; k < j; ++k) value -= G[static_cast<std::size_t>(j) * P + k] * z[k];
      z[j] = value / G[static_cast<std::size_t>(j) * P + j];
    }
    for (int j = p - 1; j >= 0; --j) {
      double value = z[j];
      for (int k = j + 1; k < p; ++k) value -= G[static_cast<std::size_t>(k) * P + j] * result[k];
      result[j] = value / G[static_cast<std::size_t>(j) * P + j];
      if (!std::isfinite(result[j])) return 4;
    }
    return 0;
  } catch (...) { return 5; }
}
