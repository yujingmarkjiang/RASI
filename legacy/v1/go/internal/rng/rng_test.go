package rng

import (
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

func TestMatchesCpp(t *testing.T) {
	bin := os.Getenv("RASI_BIN")
	if bin == "" {
		for _, c := range []string{"cpp/build/rasi", "../cpp/build/rasi", "../../cpp/build/rasi", "../../../cpp/build/rasi"} {
			if _, err := os.Stat(c); err == nil {
				bin = c
				break
			}
		}
	}
	if _, err := os.Stat(bin); err != nil {
		t.Skip("c++ engine not built")
	}
	out, err := exec.Command(bin, "rng", "1").Output()
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Fields(string(out))
	if len(lines) < 8 {
		t.Fatalf("short rng dump %q", out)
	}
	r := New(1)
	for i := 0; i < 4; i++ {
		got := r.Next()
		want, _ := strconv.ParseUint(lines[i], 10, 64)
		if got != want {
			t.Fatalf("uint64 %d got %d want %d", i, got, want)
		}
	}
	for i := 0; i < 4; i++ {
		got := r.Normal()
		want, _ := strconv.ParseFloat(lines[4+i], 64)
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("normal %d got %.17g want %.17g", i, got, want)
		}
	}
}
