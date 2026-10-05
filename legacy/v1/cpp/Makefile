CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unused-function
SRC = src/main.cpp src/engine.cpp src/gru.cpp src/infer.cpp
BIN = build/rasi

.PHONY: all test clean gpu

all: $(BIN)

$(BIN): $(SRC) src/engine.hpp src/rng.hpp src/linalg.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC)

test: $(BIN)
	./$(BIN) check
	./$(BIN) gru

clean:
	rm -rf build

gpu:
	mkdir -p build
	nvcc -O3 -std=c++17 -arch=sm_100 -c src/gru_cuda.cu -o build/gru_cuda.o
	$(CXX) $(CXXFLAGS) -DRASI_CUDA -o build/rasi-gpu $(SRC) build/gru_cuda.o -lcudart
