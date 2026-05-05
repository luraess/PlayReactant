"""
ALPS submission script for bench_sweep.jl.

Launches 4 independent Julia processes in parallel, each pinned to one GPU
(CUDA_VISIBLE_DEVICES=0..3), so all 4 GPUs on the node are exercised at once.
Results land in runs/<timestamp>_<tag>/gpu=<id>/<jobid>.out

Usage:
    julia --project bench_sweep_submit.jl
"""

using Dates, Random

username   = ENV["USER"]
account    = "c44"
time_limit = "01:30:00"
submit     = true

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

# ---- Single SLURM job, 1 GPU ------------------------------------------
sbatch    = joinpath(out_path, "submit.sh")
julia_bin = Base.julia_cmd()[1]

open(sbatch, "w") do io
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

export JULIA_CUDA_USE_COMPAT=false
export BENCH_BACKEND=gpu
export BENCH_RES_S2D=64,128,256,512,1024,2048,4096,8192
export BENCH_RES_S3D=32,64,128,256,512
export BENCH_RES_PW2D=64,128,256,512,1024,2048,4096
export BENCH_RES_PW3D=32,64,128,256,512
export XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80 \${XLA_FLAGS}"
export XLA_FLAGS="--xla_disable_hlo_passes=host-offload-legalize,hlo_constant_splitter,multi_output_fusion \${XLA_FLAGS}"
export XLA_REACTANT_GPU_MEM_FRACTION=0.85
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY
ulimit -s unlimited
ulimit -S -c0

$julia_bin --project=$project_path --startup-file=no --threads=16 --compiled-modules=strict -O0 \\
    $bench_file
""")
end

@info "Submit script: $sbatch"

if submit
    run(`sbatch $sbatch`)
    run(`squeue -u $username`)
else
    @warn "submit=false — job not submitted"
end
