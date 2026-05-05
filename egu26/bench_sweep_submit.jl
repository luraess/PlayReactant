"""
ALPS submission script for bench_sweep.jl.

Launches 4 independent Julia processes in parallel, each pinned to one GPU
(CUDA_VISIBLE_DEVICES=0..3), so all 4 GPUs on the node are exercised at once.
Results land in runs/<timestamp>_<tag>/gpu=<id>/<jobid>.out

Usage:
    julia --project egu26/bench_sweep_submit.jl
"""

using Dates, Random

username      = ENV["USER"]
account       = "c44"
time_limit    = "01:00:00"
gpus_per_node = 4
submit        = true

out_dir    = @__DIR__
run_prefix = get(ENV, "PLAYREACTANT_RUN_PREFIX",  "runs")
run_tag    = get(ENV, "PLAYREACTANT_RUN_POSTFIX", randstring(4))
timestamp  = replace(string(now(UTC)), ':' => '-')
out_path   = joinpath(out_dir, run_prefix, "$(timestamp)_$(run_tag)")
mkpath(out_path)

project_path = dirname(@__DIR__)
bench_file   = joinpath(@__DIR__, "bench_sweep.jl")
isfile(bench_file) || error("bench_sweep.jl not found at $bench_file")

# Capture repo state
cp(bench_file, joinpath(out_path, "bench_sweep.jl"))
for f in ("Project.toml", "Manifest.toml")
    src = joinpath(project_path, f)
    isfile(src) && cp(src, joinpath(out_path, f))
end

git_describe = try readchomp(`git -C $project_path --no-pager describe --tags --always --dirty`) catch; "unknown" end
git_branch   = try readchomp(`git -C $project_path rev-parse --abbrev-ref HEAD`)                catch; "unknown" end
open(joinpath(out_path, "run-info.toml"), "w") do io
    println(io, "git_describe = \"$git_describe\"")
    println(io, "git_branch   = \"$git_branch\"")
end

@info "Output directory: $out_path"

# ---- Per-GPU launcher scripts ------------------------------------------
for gpu_id in 0:(gpus_per_node - 1)
    gpu_dir = joinpath(out_path, "gpu=$gpu_id")
    mkpath(gpu_dir)

    launcher = joinpath(gpu_dir, "launcher.sh")
    open(launcher, "w") do io
        print(io, """
#!/usr/bin/env sh

export CUDA_VISIBLE_DEVICES=$gpu_id
export TZ=UTC

export JULIA_DEPOT_PATH=$(join(Base.DEPOT_PATH, ':'))

export XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80 \${XLA_FLAGS}"
export XLA_FLAGS="--xla_disable_hlo_passes=host-offload-legalize,hlo_constant_splitter,multi_output_fusion \${XLA_FLAGS}"
export XLA_REACTANT_GPU_MEM_FRACTION=0.85
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

exec "\${@}"
""")
    end
    chmod(launcher, 0o755)
end

# ---- Single SLURM job that fans out to 4 GPUs -------------------------
sbatch = joinpath(out_path, "submit.sh")
julia_bin = Base.julia_cmd()[1]

open(sbatch, "w") do io
    # Build the 4 srun lines (one per GPU, all backgrounded then waited on)
    srun_lines = join(["""
    srun --cpu-bind=sockets --mem-bind=local --exclusive \\
         --uenv="julia/25.5:v1" \\
         --view=juliaup --preserve-env \\
         $(out_path)/gpu=$gpu_id/launcher.sh \\
         $julia_bin --project=$project_path --startup-file=no --threads=16 --compiled-modules=strict -O0 \\
         $bench_file \\
         > $(out_path)/gpu=$gpu_id/\${SLURM_JOB_ID}.out 2>&1 &""" for gpu_id in 0:(gpus_per_node - 1)], "\n")

    print(io, """
#!/bin/bash -l

#SBATCH --job-name="bench_sweep"
#SBATCH --time=$time_limit
#SBATCH --output=$(out_path)/%j.out
#SBATCH --error=$(out_path)/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --account=$account
#SBATCH --exclusive

JULIA_CUDA_USE_COMPAT=false
ulimit -s unlimited
ulimit -S -c0

$srun_lines

wait
echo "All GPU benchmarks finished."
""")
end

@info "Submit script: $sbatch"

if submit
    run(`sbatch $sbatch`)
    run(`squeue -u $username`)
else
    @warn "submit=false — job not submitted"
end
