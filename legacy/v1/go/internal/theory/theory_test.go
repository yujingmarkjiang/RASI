package theory

import "math"
import "testing"

func TestWalkAndAnchor(t *testing.T) {
	obs := WalkVar(8, 5, 30, 4000, 2)
	th := WalkTheory(8, 5, 30)
	if math.Abs(obs-th)/th > 0.08 {
		t.Fatalf("walk %g vs %g", obs, th)
	}
	for _, lam := range []float64{0.3, 1, 3} {
		obs = AnchorVar(8, 5, lam, 4000, 400, 3+uint64(lam*10))
		th = AnchorTheory(8, 5, lam)
		if math.Abs(obs-th)/th > 0.12 {
			t.Fatalf("anchor λ=%g %g vs %g", lam, obs, th)
		}
	}
	b := Guarantee(0, 5.42, 15, 12, 0.05)
	if b >= 0 {
		t.Fatalf("expected a negative slack at this audit size, got %g", b)
	}
}
