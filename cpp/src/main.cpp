// The CLI delegates to the single shared controller. Statistical computation
// may use librasi.so; neither this launcher nor the linear solver claims CUDA.
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  try {
    fs::path root;
    if (const char* supplied = std::getenv("RASI_ROOT")) root = fs::absolute(supplied);
    else {
      char executable[8192];
      const auto n = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
      fs::path exe;
      if (n >= 0) { executable[n] = 0; exe = executable; }
      else exe = fs::canonical(argv[0]);
      root = exe.parent_path().parent_path().parent_path();
    }
    const auto module = root / "python";
    if (!fs::exists(module / "rasi" / "cli.py")) {
      std::cerr << "Cannot locate the RASI Python package; set RASI_ROOT to the repository root\n";
      return 2;
    }
    const char* old = std::getenv("PYTHONPATH");
    const std::string pythonpath = module.string() + (old && *old ? std::string(":") + old : "");
    if (setenv("PYTHONPATH", pythonpath.c_str(), 1) != 0) return 2;
    const char* interpreter = std::getenv("RASI_PYTHON");
    std::vector<std::string> args = {interpreter && *interpreter ? interpreter : "python3", "-m", "rasi"};
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    std::vector<char*> pointers;
    for (auto& arg : args) pointers.push_back(arg.data());
    pointers.push_back(nullptr);
    execvp(pointers[0], pointers.data());
    std::cerr << "Cannot launch Python: " << std::strerror(errno) << "\n";
    return 127;
  } catch (const std::exception& error) {
    std::cerr << "rasi: " << error.what() << "\n";
    return 2;
  }
}
