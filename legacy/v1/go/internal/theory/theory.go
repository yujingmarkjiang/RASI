package theory

import (
	"math"

	"rasi/internal/rng"
)

func WalkVar(sigma, m float64, k, reps int, seed uint64) float64 {
	r := rng.New(seed)
	th := make([]float64, reps)
	sd := sigma / math.Sqrt(m)
	for step := 0; step < k; step++ {
		for i := 0; i < reps; i++ {
			th[i] += r.Normal() * sd
		}
	}
	return popVar(th)
}

func WalkTheory(sigma, m float64, k int) float64 { return float64(k) * sigma * sigma / m }

func AnchorVar(sigma, m, lam float64, reps, burn int, seed uint64) float64 {
	r := rng.New(seed)
	th := make([]float64, reps)
	sd := sigma / math.Sqrt(m)
	for step := 0; step < burn; step++ {
		for i := 0; i < reps; i++ {
			xi := r.Normal() * sd
			th[i] = (lam*0 + th[i] + xi) / (1 + lam)
		}
	}
	return popVar(th)
}

func AnchorTheory(sigma, m, lam float64) float64 {
	return (sigma * sigma / m) / (lam * (2 + lam))
}

func Guarantee(eps, sigma float64, rounds int, audit int, delta float64) float64 {
	return eps - 2*sigma*math.Sqrt(math.Log(2*float64(rounds)/delta)/float64(audit))
}

func popVar(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	s := 0.0
	for _, x := range v {
		d := x - m
		s += d * d
	}
	return s / float64(len(v))
}
