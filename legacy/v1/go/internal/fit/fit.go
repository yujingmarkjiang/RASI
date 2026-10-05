package fit

import "math"

func SolveSPD(A, b []float64, n int) []float64 {
	L := make([]float64, n*n)
	ok := true
	for i := 0; i < n && ok; i++ {
		s := A[i*n+i]
		for k := 0; k < i; k++ {
			s -= L[i*n+k] * L[i*n+k]
		}
		if !(s > 1e-18) {
			ok = false
			break
		}
		L[i*n+i] = math.Sqrt(s)
		for j := i + 1; j < n; j++ {
			t := A[j*n+i]
			for k := 0; k < i; k++ {
				t -= L[j*n+k] * L[i*n+k]
			}
			L[j*n+i] = t / L[i*n+i]
		}
	}
	x := make([]float64, n)
	if !ok {
		return solveGEPP(A, b, n)
	}
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := b[i]
		for k := 0; k < i; k++ {
			s -= L[i*n+k] * y[k]
		}
		y[i] = s / L[i*n+i]
	}
	for i := n - 1; i >= 0; i-- {
		s := y[i]
		for k := i + 1; k < n; k++ {
			s -= L[k*n+i] * x[k]
		}
		x[i] = s / L[i*n+i]
	}
	return x
}

func solveGEPP(A, b []float64, n int) []float64 {
	M := make([]float64, n*(n+1))
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			M[i*(n+1)+j] = A[i*n+j]
		}
		M[i*(n+1)+n] = b[i]
	}
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(M[col*(n+1)+col])
		for r := col + 1; r < n; r++ {
			v := math.Abs(M[r*(n+1)+col])
			if v > best {
				best = v
				piv = r
			}
		}
		if piv != col {
			for j := col; j <= n; j++ {
				M[col*(n+1)+j], M[piv*(n+1)+j] = M[piv*(n+1)+j], M[col*(n+1)+j]
			}
		}
		d := M[col*(n+1)+col]
		for r := col + 1; r < n; r++ {
			f := M[r*(n+1)+col] / d
			M[r*(n+1)+col] = 0
			for j := col + 1; j <= n; j++ {
				M[r*(n+1)+j] -= f * M[col*(n+1)+j]
			}
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := M[i*(n+1)+n]
		for j := i + 1; j < n; j++ {
			s -= M[i*(n+1)+j] * x[j]
		}
		x[i] = s / M[i*(n+1)+i]
	}
	return x
}

type Pair struct{ P, A int }

type Env struct {
	NP, NA, NT, P, I2 int
	Floor, Thresh     float64
	LamMu, LamB, LamD float64
	Y                 []float64
	Train, Audit, Hid []Pair
	Hidden            []int
}

func (e Env) mu(a, t int) int    { return a*e.NT + t }
func (e Env) beta(p int) int     { return e.NA*e.NT + p }
func (e Env) delta(p, a int) int { return e.NA*e.NT + e.NP + (p*e.NA + a) }
func (e Env) at(p, a, t int) int { return (p*e.NA+a)*e.NT + t }

type obs struct {
	i0, i1, i2 int
	y, w       float64
}

func (e Env) addTraj(dst []obs, p, a int, ys []float64, w float64) []obs {
	for t := 0; t < e.NT; t++ {
		dst = append(dst, obs{e.mu(a, t), e.beta(p), e.delta(p, a), ys[t], w})
	}
	return dst
}

func (e Env) traj(p, a int) []float64 {
	ys := make([]float64, e.NT)
	for t := 0; t < e.NT; t++ {
		ys[t] = e.Y[e.at(p, a, t)]
	}
	return ys
}

func (e Env) Fit(pairs []Pair, w float64) []float64 {
	var os []obs
	for _, pa := range pairs {
		os = e.addTraj(os, pa.P, pa.A, e.traj(pa.P, pa.A), w)
	}
	return e.fitObs(os)
}

func (e Env) fitObs(os []obs) []float64 {
	P := e.P
	if P == 0 {
		P = e.NA*e.NT + e.NP + e.NP*e.NA
	}
	G := make([]float64, P*P)
	b := make([]float64, P)
	for _, o := range os {
		ix := [3]int{o.i0, o.i1, o.i2}
		for a := 0; a < 3; a++ {
			b[ix[a]] += o.w * o.y
			G[ix[a]*P+ix[a]] += o.w
			for c := a + 1; c < 3; c++ {
				G[ix[a]*P+ix[c]] += o.w
				G[ix[c]*P+ix[a]] += o.w
			}
		}
	}
	for i := 0; i < e.NA*e.NT; i++ {
		G[i*P+i] += e.LamMu
	}
	for p := 0; p < e.NP; p++ {
		G[e.beta(p)*P+e.beta(p)] += e.LamB
	}
	for p := 0; p < e.NP; p++ {
		for a := 0; a < e.NA; a++ {
			d := e.delta(p, a)
			G[d*P+d] += e.LamD
		}
	}
	return SolveSPD(G, b, P)
}

func (e Env) Pred(th []float64, p, a, t int) float64 {
	return th[e.mu(a, t)] + th[e.beta(p)] + th[e.delta(p, a)]
}
