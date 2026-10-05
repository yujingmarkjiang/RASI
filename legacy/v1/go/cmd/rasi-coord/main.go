package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

type config struct {
	Task     string  `json:"task"`
	GPU      string  `json:"gpu"`
	Devices  int     `json:"devices"`
	Parallel string  `json:"parallel"`
	Env      string  `json:"env"`
	Masks    int     `json:"masks"`
	Seeds    int     `json:"seeds"`
	Rounds   int     `json:"rounds"`
	M        int     `json:"m"`
	Lambda   float64 `json:"lambda"`
	Model    string  `json:"model"`
	Cohort   string  `json:"cohort"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: rasi-coord plan|train|test|infer --config FILE [--cpu-mesh N] [--masks N] [--rounds N]")
		os.Exit(2)
	}
	cmd := os.Args[1]
	cfgPath := flag("--config", "rasi/configs/train_32xb200.json")
	cpuMesh := 0
	if v := flag("--cpu-mesh", ""); v != "" {
		cpuMesh, _ = strconv.Atoi(v)
	}
	cfg := config{GPU: "B200", Devices: 32, Parallel: "mask", Env: "qt", Masks: 32, Seeds: 1, Rounds: 15, M: 5, Lambda: 1, Model: "linear"}
	if raw, err := os.ReadFile(cfgPath); err == nil {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			fatal("config: %v", err)
		}
	} else if flag("--config", "") != "" {
		fatal("cannot read %s", cfgPath)
	}
	if v := flag("--masks", ""); v != "" {
		cfg.Masks, _ = strconv.Atoi(v)
	}
	if v := flag("--rounds", ""); v != "" {
		cfg.Rounds, _ = strconv.Atoi(v)
	}
	if v := flag("--seeds", ""); v != "" {
		cfg.Seeds, _ = strconv.Atoi(v)
	}
	cohort := resolveCohort(cfgPath, cfg.Cohort)
	switch cmd {
	case "plan":
		printPlan(cfg, cpuMesh, cohort)
	case "train":
		cfg.Task = "train"
		printPlan(cfg, cpuMesh, cohort)
		if err := requireDevices(cfg.Devices, cpuMesh, "train"); err != nil {
			fatal("%v", err)
		}
		n := cfg.Devices
		if cpuMesh > 0 {
			n = cpuMesh
		}
		runTrain(cfg, cohort, n)
	case "test":
		cfg.Task = "test"
		if cfg.Masks > 4 {
			cfg.Masks = 2
		}
		if cfg.Rounds > 8 {
			cfg.Rounds = 6
		}
		printPlan(cfg, cpuMesh, cohort)
		runTest(cfg, cohort)
	case "infer":
		if cfg.Devices == 0 {
			cfg.Devices = 2
		}
		printPlan(cfg, cpuMesh, cohort)
		if err := requireDevices(cfg.Devices, cpuMesh, "infer"); err != nil {
			fatal("%v", err)
		}
		n := cfg.Devices
		if cpuMesh > 0 {
			n = cpuMesh
		}
		if n != 2 && cpuMesh == 0 {
			fatal("inference is a 2-device shard, config asks for %d", n)
		}
		runInfer(cfg, cohort, n)
	default:
		fatal("unknown command %s", cmd)
	}
}

func isTTY() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func resolveCohort(cfgPath, cohort string) string {
	if strings.TrimSpace(cohort) == "" {
		return ""
	}
	if filepath.IsAbs(cohort) {
		return cohort
	}
	if _, err := os.Stat(cohort); err == nil {
		return cohort
	}
	cfgDir := filepath.Dir(cfgPath)
	root := filepath.Dir(cfgDir)
	cands := []string{
		filepath.Join(cfgDir, cohort),
		filepath.Join(root, strings.TrimPrefix(cohort, "rasi/")),
		filepath.Join(filepath.Dir(root), cohort),
	}
	for _, c := range cands {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return cohort
}

func printPlan(cfg config, cpu int, cohort string) {
	n := cfg.Devices
	mode := "gpu"
	if cpu > 0 {
		mode = "cpu-mesh"
		n = cpu
	}
	if !isTTY() {
		fmt.Printf("{\"task\":%q,\"gpu\":%q,\"devices\":%d,\"mode\":%q,\"parallel\":%q,\"model\":%q,\"env\":%q,\"cohort\":%q,\"masks\":%d,\"rounds\":%d}\n",
			cfg.Task, cfg.GPU, n, mode, cfg.Parallel, cfg.Model, cfg.Env, cohort, cfg.Masks, cfg.Rounds)
		return
	}
	label := cfg.Task
	if label == "" {
		label = "plan"
	}
	fmt.Printf("\n\033[1mRASI  %s\033[0m\n", label)
	if cpu > 0 {
		fmt.Printf("  mesh     %d processes    target %d× %s\n", n, cfg.Devices, cfg.GPU)
	} else {
		fmt.Printf("  mesh     %d× %s    %s\n", n, cfg.GPU, cfg.Parallel)
	}
	if cohort != "" {
		fmt.Printf("  cohort   %s\n", cohort)
	} else {
		fmt.Printf("  cohort   simulated %s\n", cfg.Env)
	}
	if cfg.Task == "infer" {
		fmt.Printf("  serve    best accepted round, then shard means | individuals\n")
	} else {
		fmt.Printf("  loop     %d masks    %d rounds    %d rollouts    λ %.3g\n", cfg.Masks, cfg.Rounds, cfg.M, cfg.Lambda)
	}
}

func requireDevices(want, cpu int, task string) error {
	if cpu > 0 {
		return nil
	}
	got := visibleGPUs()
	if got != want {
		return fmt.Errorf("%s expects %d× B200 (visible %d). On a CPU box pass --cpu-mesh %d to run the same mesh", task, want, got, map[string]int{"train": 32, "infer": 2}[task])
	}
	return nil
}

func visibleGPUs() int {
	if v, ok := os.LookupEnv("CUDA_VISIBLE_DEVICES"); ok {
		v = strings.TrimSpace(v)
		if v == "" || v == "-1" {
			return 0
		}
		return len(strings.Split(v, ","))
	}
	out, err := exec.Command("nvidia-smi", "-L").Output()
	if err != nil {
		return 0
	}
	n := 0
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "GPU ") {
			n++
		}
	}
	return n
}

func rasiBin() string {
	if p := os.Getenv("RASI_BIN"); p != "" {
		return p
	}
	var cands []string
	if exe, err := os.Executable(); err == nil {
		cands = append(cands, filepath.Clean(filepath.Join(filepath.Dir(exe), "../../cpp/build/rasi")))
	}
	cands = append(cands,
		"cpp/build/rasi",
		"./cpp/build/rasi",
	)
	for _, c := range cands {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return "rasi"
}

func flag(name, dflt string) string {
	for i := 0; i < len(os.Args)-1; i++ {
		if os.Args[i] == name {
			return os.Args[i+1]
		}
	}
	return dflt
}

type cond struct {
	CCIK   float64   `json:"cci_K"`
	Kept   float64   `json:"kept"`
	N      float64   `json:"n"`
	Hidden []float64 `json:"hidden"`
}

type acc struct{ cci, kept, n float64 }

func runTrain(cfg config, cohort string, world int) {
	dir, err := os.MkdirTemp("", "rasi-train-")
	if err != nil {
		fatal("%v", err)
	}
	defer os.RemoveAll(dir)
	bins := rasiBin()
	errCh := make(chan error, world)
	for rank := 0; rank < world; rank++ {
		rank := rank
		go func() {
			out := filepath.Join(dir, fmt.Sprintf("rank%d.json", rank))
			args := []string{
				"train",
				"--format", "json",
				"--masks", strconv.Itoa(cfg.Masks),
				"--seeds", strconv.Itoa(cfg.Seeds),
				"--rounds", strconv.Itoa(cfg.Rounds),
				"--m", strconv.Itoa(cfg.M),
				"--lambda", fmt.Sprint(cfg.Lambda),
				"--rank", strconv.Itoa(rank),
				"--world-size", strconv.Itoa(world),
				"--trace", "0",
				"--task", "train",
				"--out", out,
			}
			if cohort != "" {
				args = append(args, "--cohort", cohort)
			} else {
				args = append(args, "--env", cfg.Env)
			}
			cmd := exec.Command(bins, args...)
			cmd.Env = append(os.Environ(), fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", rank))
			msg, err := cmd.CombinedOutput()
			if err != nil {
				errCh <- fmt.Errorf("rank %d: %v %s", rank, err, msg)
				return
			}
			errCh <- nil
		}()
	}
	for rank := 0; rank < world; rank++ {
		if err := <-errCh; err != nil {
			fatal("%v", err)
		}
	}
	sum := map[string]*acc{}
	for rank := 0; rank < world; rank++ {
		raw, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("rank%d.json", rank)))
		if err != nil {
			fatal("%v", err)
		}
		var doc struct {
			Conditions map[string]cond `json:"conditions"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			fatal("rank json: %v", err)
		}
		for name, c := range doc.Conditions {
			a := sum[name]
			if a == nil {
				a = &acc{}
				sum[name] = a
			}
			w := c.N
			if w == 0 {
				w = 1
			}
			a.cci += c.CCIK * w
			a.kept += c.Kept * w
			a.n += w
		}
	}
	if isTTY() {
		fmt.Printf("\n  \033[2m%-12s %10s %8s\033[0m\n", "condition", "cci", "kept")
		order := []string{"m0", "standard", "rsi", "rasi"}
		seen := map[string]bool{}
		for _, name := range order {
			if sum[name] != nil {
				seen[name] = true
				printCond(name, sum[name])
			}
		}
		for name, a := range sum {
			if !seen[name] {
				printCond(name, a)
			}
		}
		fmt.Println()
		return
	}
	fmt.Printf("{\"merged\":{")
	first := true
	for name, a := range sum {
		if !first {
			fmt.Print(",")
		}
		first = false
		fmt.Printf("%q:{\"cci_K\":%.6g,\"kept\":%.6g,\"n\":%.0f}", name, a.cci/a.n, a.kept/a.n, a.n)
	}
	fmt.Printf("},\"world\":%d,\"gpu\":\"%s\",\"parallel\":\"mask\",\"cohort\":%q}\n", world, cfg.GPU, cohort)
}

func printCond(name string, a *acc) {
	color := ""
	if name == "rasi" {
		color = "\033[32m"
	} else if name == "standard" {
		color = "\033[31m"
	}
	fmt.Printf("%s  %-12s %+10.3f %7.0f%%\033[0m\n", color, name, a.cci/a.n, 100*a.kept/a.n)
}

func runTest(cfg config, cohort string) {
	if cohort == "" {
		fatal("test needs a cohort in the config")
	}
	bins := rasiBin()
	cmd := exec.Command(bins, "test",
		"--cohort", cohort,
		"--task", "test",
		"--masks", strconv.Itoa(cfg.Masks),
		"--seeds", "1",
		"--rounds", strconv.Itoa(cfg.Rounds),
		"--m", strconv.Itoa(cfg.M),
		"--lambda", fmt.Sprint(cfg.Lambda),
		"--trace", "0",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fatal("test: %v", err)
	}
}

func runInfer(cfg config, cohort string, devices int) {
	bins := rasiBin()
	dir, _ := os.MkdirTemp("", "rasi-infer-")
	defer os.RemoveAll(dir)
	ckpt := filepath.Join(dir, "ckpt.json")
	var fit *exec.Cmd
	if cohort != "" {
		fit = exec.Command(bins, "loop", "--cohort", cohort, "--task", "apply", "--format", "json",
			"--masks", "1", "--seeds", "1", "--rounds", "4", "--m", "4", "--lambda", "1",
			"--trace", "0", "--condition", "rasi", "--out", ckpt, "--seed", "11")
	} else {
		fit = exec.Command(bins, "loop", "--env", "tiny", "--masks", "1", "--seeds", "1", "--rounds", "0",
			"--trace", "0", "--condition", "m0", "--format", "json", "--out", ckpt, "--seed", "1")
	}
	if msg, err := fit.CombinedOutput(); err != nil {
		fatal("checkpoint: %v %s", err, msg)
	}
	enc := exec.Command(bins, "infer", "--ckpt", ckpt, "--stage", "encode", "--device", "0", "--devices", strconv.Itoa(devices),
		"--route", "shard", "--p", "1", "--a", "0", "--t", "2")
	enc.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=0")
	encOut, err := enc.Output()
	if err != nil {
		fatal("encode: %v", err)
	}
	var encDoc struct {
		Part float64 `json:"part"`
	}
	if err := json.Unmarshal(encOut, &encDoc); err != nil {
		fatal("encode json: %v %s", err, encOut)
	}
	roll := exec.Command(bins, "infer", "--ckpt", ckpt, "--stage", "rollout", "--device", "1", "--devices", strconv.Itoa(devices),
		"--route", "shard", "--p", "1", "--a", "0", "--t", "2", "--part", fmt.Sprint(encDoc.Part))
	roll.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=1")
	rollOut, err := roll.Output()
	if err != nil {
		fatal("rollout: %v %s", err, rollOut)
	}
	var rollDoc struct {
		Prediction float64 `json:"prediction"`
	}
	if err := json.Unmarshal(rollOut, &rollDoc); err != nil {
		fatal("rollout json: %v", err)
	}
	full := exec.Command(bins, "infer", "--ckpt", ckpt, "--stage", "full", "--device", "0", "--devices", strconv.Itoa(devices),
		"--p", "1", "--a", "0", "--t", "2")
	fullOut, err := full.Output()
	if err != nil {
		fatal("full: %v", err)
	}
	var fullDoc struct {
		Prediction float64 `json:"prediction"`
	}
	if err := json.Unmarshal(fullOut, &fullDoc); err != nil {
		fatal("full json: %v", err)
	}
	diff := rollDoc.Prediction - fullDoc.Prediction
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-6 {
		fatal("sharded inference drifted by %g", diff)
	}
	if isTTY() {
		fmt.Printf("\n\033[1mRASI  infer\033[0m\n")
		if cohort != "" {
			fmt.Printf("  cohort      %s\n", cohort)
		}
		fmt.Printf("  route       device 0 encodes the drug-time mean, device 1 adds the person\n")
		fmt.Printf("\033[1;32m  prediction  %.2f ms\033[0m\n", rollDoc.Prediction)
		fmt.Printf("  check       full path %.2f ms    absolute difference %.3g\n\n", fullDoc.Prediction, diff)
		return
	}
	fmt.Printf("{\"infer_devices\":%d,\"gpu\":\"B200\",\"route\":\"shard\",\"prediction\":%.8g,\"full\":%.8g,\"abs_diff\":%.3g}\n",
		devices, rollDoc.Prediction, fullDoc.Prediction, diff)
}

func fatal(f string, args ...any) {
	fmt.Fprintf(os.Stderr, f+"\n", args...)
	os.Exit(1)
}
