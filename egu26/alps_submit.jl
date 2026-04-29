"""
ALPS submission script for diff2D_flux_shard.jl.

Usage:
    julia --project alps_submit.jl diff2D_flux_shard.jl

Modelled after GB-25/sharding/alps_scaling_test.jl +
common_submission_generator.jl.
"""

using Dates, Random

username      = ENV["USER"]
account       = "c44"           # ALPS project account
run_name      = "diff2D_shard_"
time_limit    = "00:30:00"
gpus_per_node = 4
cpus_per_task = 288
submit        = true            # set to true to actually sbatch
out_dir       = @__DIR__

run_prefix  = get(ENV, "PLAYREACTANT_RUN_PREFIX",  "runs")
run_postfix = get(ENV, "PLAYREACTANT_RUN_POSTFIX", randstring(4))

# GPU counts to sweep — must be perfect squares (mesh_factors requires D*D == N)
# and multiples of gpus_per_node.
Ngpus = [4, 16]

for N in Ngpus
    isqrt(N)^2 == N        || error("Ngpus entry $N is not a perfect square")
    N % gpus_per_node == 0 || error("Ngpus entry $N is not a multiple of gpus_per_node=$gpus_per_node")
end
@info "GPU counts to submit: $Ngpus"

# ---- Input file --------------------------------------------------------
if length(Base.ARGS) != 1
    error("""
          Usage:
              julia $(basename(@__FILE__)) <SCRIPT_PATH>
          E.g.:
              julia $(basename(@__FILE__)) diff2D_flux_shard.jl
          """)
end

input_file = joinpath(@__DIR__, Base.ARGS[1])
isfile(input_file) || error("File $input_file does not exist")

timestamp = replace(string(now(UTC)), ':' => '-')
out_path  = joinpath(out_dir, run_prefix, "$(timestamp)_$(run_postfix)")
mkpath(out_path)

project_path = dirname(@__DIR__)   # PlayReactant root (has Project.toml)

# Capture repo state
run_file = joinpath(out_path, basename(input_file))
cp(input_file, run_file)
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

# ---- Per-GPU-count jobs ------------------------------------------------
for Ngpu in Ngpus
    Nnodes     = ceil(Int, Ngpu / gpus_per_node)
    ngpu_str   = lpad(Ngpu, 5, '0')
    job_dir    = joinpath(out_path, "ngpu=$(ngpu_str)")
    mkpath(job_dir)

    job_name = "$(run_name)$(Dates.format(now(UTC), "ud"))_ngpu$(ngpu_str)"

    # launcher.sh — sets env vars, then exec's Julia
    launcher = joinpath(job_dir, "launcher.sh")
    open(launcher, "w") do io
        print(io, """
#!/usr/bin/env sh

export CUDA_VISIBLE_DEVICES=$(join(0:(min(Ngpu, gpus_per_node) - 1), ','))
export TZ=UTC
export Ngpu=$Ngpu

export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=$(join(Base.DEPOT_PATH, ':'))

# Avoid XLA collective timeouts
export XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80 \${XLA_FLAGS}"
export XLA_FLAGS="--xla_disable_hlo_passes=host-offload-legalize,hlo_constant_splitter,multi_output_fusion \${XLA_FLAGS}"
# Leave headroom for collective scratch buffers and rematerialization (especially at 16+ GPUs)
export XLA_REACTANT_GPU_MEM_FRACTION=0.75
export XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=95 \${XLA_FLAGS}"

# Ensure Julia's bundled OpenSSL is found before system OpenSSL
# export LD_LIBRARY_PATH="/capstor/scratch/cscs/lraess/julia_local/artifacts/bae3e8f87928cd4450404cb879640f42e960d065/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

# Unset proxies — XLA can hang indefinitely if these are set
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

exec "\${@}"
echo "[\${SLURM_JOB_ID}.\${SLURM_PROCID}] Process exited with code \${?}"
""")
    end
    chmod(launcher, 0o755)

    # submit.sh — the actual SLURM batch script
    sbatch = joinpath(job_dir, "submit.sh")
    open(sbatch, "w") do io
        print(io, """
#!/bin/bash -l

#SBATCH --job-name="$job_name"
#SBATCH --time=$time_limit
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --nodes=$Nnodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$gpus_per_node
#SBATCH --gpu-bind=per_task:$gpus_per_node
#SBATCH --constraint=gpu
#SBATCH --account=$account
#SBATCH --exclusive

JULIA_CUDA_USE_COMPAT=false

ulimit -s unlimited
ulimit -S -c0   # disable core dumps

srun --uenv="/iopsstor/scratch/cscs/omlins/uenv_julia/julia_26_3_v1_gh200.squashfs" \\
     --view=juliaup \\
     --preserve-env \\
     $(job_dir)/launcher.sh \\
     $(Base.julia_cmd()[1]) --project=$project_path --startup-file=no --threads=16 --compiled-modules=strict -O0 \\
     $run_file
""")
    end

    @info "Ngpu=$Ngpu  nodes=$Nnodes  →  $sbatch"

    if submit
        run(`sbatch $sbatch`)
        run(`squeue -u $username`)
    else
        @warn "submit=false — job not submitted"
    end
end
