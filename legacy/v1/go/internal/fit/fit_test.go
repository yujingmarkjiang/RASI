package fit

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"testing"
)

func TestSolveResidual(t *testing.T) {
	A := []float64{4, 1, 0, 1, 3, 1, 0, 1, 2}
	b := []float64{1, 2, 3}
	x := SolveSPD(A, b, 3)
	r0 := 4*x[0] + x[1] - 1
	r1 := x[0] + 3*x[1] + x[2] - 2
	r2 := x[1] + 2*x[2] - 3
	if math.Abs(r0)+math.Abs(r1)+math.Abs(r2) > 1e-8 {
		t.Fatalf("residual %g %g %g", r0, r1, r2)
	}
}

func TestThetaMatchesCpp(t *testing.T) {
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
	path := t.TempDir() + "/env.json"
	cmd := exec.Command(bin, "dump", "--env", "tiny", "--seed", "1", "--out", path)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("dump: %v %s", err, out)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		NP, NA, NT, P, I2 int
		Floor, Threshold  float64
		LamMu, LamB, LamD float64
		Y, Theta0         []float64
		Train             [][]int
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	env := Env{
		NP: doc.NP, NA: doc.NA, NT: doc.NT, P: doc.P, I2: doc.I2,
		Floor: doc.Floor, Thresh: doc.Threshold,
		LamMu: doc.LamMu, LamB: doc.LamB, LamD: doc.LamD,
		Y: doc.Y,
	}
	for _, pa := range doc.Train {
		env.Train = append(env.Train, Pair{pa[0], pa[1]})
	}
	th := env.Fit(env.Train, 1)
	worst := 0.0
	for i := range th {
		worst = math.Max(worst, math.Abs(th[i]-doc.Theta0[i]))
	}
	if worst > 1e-6 {
		t.Fatalf("theta mismatch %g", worst)
	}
}
