// rasi-coord is a process supervisor, not a fictitious GPU or mask mesh.
// All scientific decisions are made by the shared RASI controller.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: rasi-coord <rasi-command> [RASI CLI arguments]")
		os.Exit(2)
	}
	binary := os.Getenv("RASI_BIN")
	if binary == "" {
		executable, err := os.Executable()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		binary = filepath.Clean(filepath.Join(filepath.Dir(executable), "../../cpp/build/rasi"))
	}
	if info, err := os.Stat(binary); err != nil || info.IsDir() {
		fmt.Fprintln(os.Stderr, "RASI launcher not found: run make or set RASI_BIN")
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	command := exec.CommandContext(ctx, binary, os.Args[1:]...)
	command.Env = os.Environ()
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := command.Run(); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) && exit.ExitCode() >= 0 {
			os.Exit(exit.ExitCode())
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
