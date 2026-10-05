package rng

import "math"

type Rng struct {
	s       [4]uint64
	spareOk bool
	spare   float64
}

func splitmix(x *uint64) uint64 {
	*x += 0x9E3779B97F4A7C15
	z := *x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func rotl(x uint64, k int) uint64 { return (x << k) | (x >> (64 - k)) }

func New(seed uint64) *Rng {
	r := &Rng{}
	x := seed
	for i := 0; i < 4; i++ {
		r.s[i] = splitmix(&x)
	}
	return r
}

func (r *Rng) Next() uint64 {
	result := rotl(r.s[1]*5, 7) * 9
	t := r.s[1] << 17
	r.s[2] ^= r.s[0]
	r.s[3] ^= r.s[1]
	r.s[1] ^= r.s[2]
	r.s[0] ^= r.s[3]
	r.s[2] ^= t
	r.s[3] = rotl(r.s[3], 45)
	return result
}

func (r *Rng) Int(n int) int {
	if n <= 1 {
		return 0
	}
	bound := uint64(n)
	limit := (^uint64(0) / bound) * bound
	for {
		v := r.Next()
		if v < limit {
			return int(v % bound)
		}
	}
}

func (r *Rng) Uniform() float64 { return float64(r.Next()>>11) * 0x1p-53 }

func (r *Rng) Normal() float64 {
	if r.spareOk {
		r.spareOk = false
		return r.spare
	}
	u1 := r.Uniform()
	if u1 < 1e-12 {
		u1 = 1e-12
	}
	u2 := r.Uniform()
	rad := math.Sqrt(-2 * math.Log(u1))
	const pi2 = 6.28318530717958647692
	r.spare = rad * math.Sin(pi2*u2)
	r.spareOk = true
	return rad * math.Cos(pi2*u2)
}
