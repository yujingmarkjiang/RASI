CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all native clean
all: build/rasi build/librasi.so
native: build/librasi.so
build/rasi: src/main.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@
build/librasi.so: src/solver.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o $@
clean:
	rm -rf build
