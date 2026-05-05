# Resolution sweep for all 4 Reactant physics solvers.
#
# Usage (from repo root):
#   julia --project bench_sweep.jl
#
# Results are saved to ./output/bench_sweep_results.toml (and .png if BENCH_VIZ=1).
#
# Environment variables:
#   BENCH_BACKEND      backend to use: gpu (default), cpu, tpu
#   BENCH_VIZ          set to 1 to save a CairoMakie figure alongside the TOML
#   BENCH_SWEEP_OUTDIR output directory (default: ./output/)
#   BENCH_SWEEP_OUT    full path to output TOML (overrides BENCH_SWEEP_OUTDIR)
#   BENCH_RES_S2D      comma-separated nx list for Stokes 2D      (default: 64,128,256,512,1024,2048,4096,8192)
#   BENCH_RES_S3D      comma-separated nx list for Stokes 3D      (default: 32,64,128,256,512)
#   BENCH_RES_PW2D     comma-separated nx list for PorowavesStokes 2D (default: 64,128,256,512,1024,2048,4096)
#   BENCH_RES_PW3D     comma-separated nx list for PorowavesStokes 3D (default: 32,64,128,256,512)
#
# Examples:
#   # Quick smoke-test on CPU
#   BENCH_BACKEND=auto BENCH_RES_S2D=64,128 BENCH_RES_S3D=32 BENCH_RES_PW2D=64 BENCH_RES_PW3D=32 \
#       julia --project bench_sweep.jl
#
#   # Full GPU sweep with figure
#   BENCH_VIZ=1 julia --project bench_sweep.jl
#
#   # Re-plot an existing result without re-running
#   julia --project bench_viz.jl
using Printf, Dates

const _bench_sweep = true   # must be in Main before any include

println("Loading solvers (this compiles Julia code, not Reactant)...")
module S2D;  include("Stokes_react2D.jl");           end
module S3D;  include("Stokes_react3D.jl");           end
module PW2D; include("PorowavesStokes_react2D.jl");  end
module PW3D; include("PorowavesStokes_react3D.jl");  end

# ── configurable via env vars ─────────────────────────────────────────
backend = Symbol(get(ENV, "BENCH_BACKEND", "auto"))  # :gpu | :tpu | :cpu
do_viz  = get(ENV, "BENCH_VIZ", "0") == "1"        # set BENCH_VIZ=1 to produce figures

_parse_res(env, default) = haskey(ENV, env) ? parse.(Int, split(ENV[env], ',')) : default
res_s2d  = _parse_res("BENCH_RES_S2D",  [64, 128])#[64, 128, 256, 512, 1024, 2048, 4096, 8192])
res_s3d  = _parse_res("BENCH_RES_S3D",  [32, 64])#[32, 64, 128, 256, 512])
res_pw2d = _parse_res("BENCH_RES_PW2D", [64, 128])#[64, 128, 256, 512, 1024, 2048, 4096])
res_pw3d = _parse_res("BENCH_RES_PW3D", [32, 64])#[32, 64, 128, 256, 512])

# ─────────────────────────────────────────────────────────
# Stokes 2D  (14 arrays per iteration, ~nx × ny each)
# ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Stokes 2D")
println("="^60)
results_s2d = []
for nx in res_s2d
# default: [512,1024,2048,4096]  |  override: BENCH_RES_S2D=64,128,256
    @printf "--- nx=%d ---\n" nx; flush(stdout)
    r = S2D.main(nx=nx, ny=nx, backend=backend, bench=true, do_plot=false, verbose=false)
    push!(results_s2d, (nx=nx, r...))
end

# ─────────────────────────────────────────────────────────
# Stokes 3D  (24 arrays per iteration, ~nx × ny × nz each)
# ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Stokes 3D")
println("="^60)
results_s3d = []
for nx in res_s3d
# default: [128,256,512,1024]  |  override: BENCH_RES_S3D=32,64,128
    @printf "--- nx=%d ---\n" nx; flush(stdout)
    r = S3D.main(nx=nx, ny=nx, nz=nx, backend=backend, bench=true, do_plot=false, verbose=false)
    push!(results_s3d, (nx=nx, r...))
end

# ─────────────────────────────────────────────────────────
# PorowavesStokes 2D  (33 arrays per iteration, ~nx × ny each)
# ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("PorowavesStokes 2D")
println("="^60)
results_pw2d = []
for nx in res_pw2d
# default: [256,512,1024,2048]  |  override: BENCH_RES_PW2D=32,64,128,256
    @printf "--- nx=%d ---\n" nx; flush(stdout)
    r = PW2D.main(nx=nx, nt=1, backend=backend, bench=true, do_plot=false, verbose=false)
    push!(results_pw2d, (nx=nx, r...))
end

# ─────────────────────────────────────────────────────────
# PorowavesStokes 3D  (44 arrays per iteration, ~nx × ny × nz each)
# ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("PorowavesStokes 3D")
println("="^60)
results_pw3d = []
for nx in res_pw3d
# default: [64,128,256,512]  |  override: BENCH_RES_PW3D=16,32,64,128
    @printf "--- nx=%d ---\n" nx; flush(stdout)
    r = PW3D.main(nx=nx, nt=1, backend=backend, bench=true, do_plot=false, verbose=false)
    push!(results_pw3d, (nx=nx, r...))
end

# ─────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
@printf "%-20s %6s %12s %10s %8s %10s\n" "solver" "nx" "t_compile[s]" "t_run[s]" "niter" "T_eff[GB/s]"
println("-"^70)
all_results = [("Stokes2D",   results_s2d),
               ("Stokes3D",   results_s3d),
               ("PorWaves2D", results_pw2d),
               ("PorWaves3D", results_pw3d)]
for (label, results) in all_results
    for r in results
        @printf "%-20s %6d %12.2f %10.3f %8d %10.2f\n" label r.nx r.t_compile r.t_run r.niter r.T_eff
    end
end

# ─────────────────────────────────────────────────────────
# Save results to TOML
# ─────────────────────────────────────────────────────────
_out_dir = get(ENV, "BENCH_SWEEP_OUTDIR", joinpath(@__DIR__, "output"))
mkpath(_out_dir)
out_file = get(ENV, "BENCH_SWEEP_OUT", joinpath(_out_dir, "bench_sweep_results.toml"))
open(out_file, "w") do io
    println(io, "# bench_sweep results — $(string(Dates.now()))")
    println(io, "backend = \"$(string(backend))\"")
    println(io)
    for (label, results) in all_results
        for r in results
            println(io, "[[runs]]")
            println(io, "solver    = \"$label\"")
            println(io, "nx        = $(r.nx)")
            println(io, "t_compile = $(r.t_compile)")
            println(io, "t_run     = $(r.t_run)")
            println(io, "niter     = $(r.niter)")
            println(io, "T_eff     = $(r.T_eff)")
            println(io)
        end
    end
end
println("\nResults saved to: $out_file")

# ─────────────────────────────────────────────────────────
# Visualisation (do_viz=true or BENCH_VIZ=1)
# ─────────────────────────────────────────────────────────
do_viz && include(joinpath(@__DIR__, "bench_viz.jl"))
