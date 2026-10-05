.PHONY: all test clean

all:
	$(MAKE) -C cpp
	mkdir -p go/build
	cd go && go build -o build/rasi-coord ./cmd/rasi-coord

test: all
	./cpp/build/rasi check
	./cpp/build/rasi gru

clean:
	$(MAKE) -C cpp clean
	rm -rf go/build
