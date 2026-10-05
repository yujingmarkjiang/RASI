#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

inline std::vector<double> solve_spd(const std::vector<double>& A, const std::vector<double>& b, int n) {
  std::vector<double> L(n * n, 0.0);
  bool ok = true;
  for (int i = 0; i < n && ok; i++) {
    double s = A[i * n + i];
    for (int k = 0; k < i; k++) s -= L[i * n + k] * L[i * n + k];
    if (!(s > 1e-18)) {
      ok = false;
      break;
    }
    L[i * n + i] = std::sqrt(s);
    for (int j = i + 1; j < n; j++) {
      double t = A[j * n + i];
      for (int k = 0; k < i; k++) t -= L[j * n + k] * L[i * n + k];
      L[j * n + i] = t / L[i * n + i];
    }
  }
  std::vector<double> x(n, 0.0);
  if (ok) {
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++) {
      double s = b[i];
      for (int k = 0; k < i; k++) s -= L[i * n + k] * y[k];
      y[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
      double s = y[i];
      for (int k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
      x[i] = s / L[i * n + i];
    }
    return x;
  }
  std::vector<double> M(n * (n + 1), 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) M[i * (n + 1) + j] = A[i * n + j];
    M[i * (n + 1) + n] = b[i];
  }
  for (int col = 0; col < n; col++) {
    int piv = col;
    double best = std::fabs(M[col * (n + 1) + col]);
    for (int r = col + 1; r < n; r++) {
      double v = std::fabs(M[r * (n + 1) + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > 0)) throw std::runtime_error("singular normal equations");
    if (piv != col) {
      for (int j = col; j <= n; j++) std::swap(M[col * (n + 1) + j], M[piv * (n + 1) + j]);
    }
    double d = M[col * (n + 1) + col];
    for (int r = col + 1; r < n; r++) {
      double f = M[r * (n + 1) + col] / d;
      M[r * (n + 1) + col] = 0;
      for (int j = col + 1; j <= n; j++) M[r * (n + 1) + j] -= f * M[col * (n + 1) + j];
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    double s = M[i * (n + 1) + n];
    for (int j = i + 1; j < n; j++) s -= M[i * (n + 1) + j] * x[j];
    x[i] = s / M[i * (n + 1) + i];
  }
  return x;
}
