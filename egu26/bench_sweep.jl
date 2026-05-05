# Resolution sweep for all 4 Reactant physics solvers.
# Run from the repo root: julia --project scripts/bench_sweep.jl
#
# Each script is wrapped in its own module to avoid namespace clashes.
# The `_bench_sweep` flag suppresses the hardcoded main() call at the
# bottom of each included script.
using Printf, Dates

const _bench_sweep = true   # must be in Main before any include

println("Loading solvers (this compiles Julia code, not Reactant)...")
module S2D;  include("Stokes_react2D.jl");           end
module S3D;  include("Stokes_react3D.jl");           end
module PW2D; include("PorowavesStokes_react2D.jl");  end
module PW3D; include("PorowavesStokes_react3D.jl");  end

# ── configurable via env vars ─────────────────────────────────────────
backend = Symbol(get(ENV, "BENCH_BACKEND", "gpu"))  # :gpu | :tpu | :cpu
do_viz  = get(ENV, "BENCH_VIZ", "0") == "1"        # set BENCH_VIZ=1 to produce figures

_parse_res(env, default) = haskey(ENV, env) ? parse.(Int, split(ENV[env], ',')) : default
res_s2d  = _parse_res("BENCH_RES_S2D",  [64, 128])#, 256, 512, 1024, 2048, 4096])
res_s3d  = _parse_res("BENCH_RES_S3D",  [32, 64])#, 128, 256, 512])
res_pw2d = _parse_res("BENCH_RES_PW2D", [64, 128])#, 256, 512, 1024, 2048])
res_pw3d = _parse_res("BENCH_RES_PW3D", [32, 64])#, 128, 256, 512])

# ─────────────────────────────────────────────────────────
# Stokes 2D  (14 arrays per iteration, ~nx × ny each)
# ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Stokes 2D")
println("="^60)
results_s2d = []
for nx in res_s2d
# default: [512,1024,2048,4096]  |  override: BENCH_RES_S2D=64,128,256
    @printf "--- nx=%d ---\n" nx
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
    @printf "--- nx=%d ---\n" nx
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
    @printf "--- nx=%d ---\n" nx
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
    @printf "--- nx=%d ---\n" nx
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
out_file = get(ENV, "BENCH_SWEEP_OUT", joinpath(@__DIR__, "bench_sweep_results.toml"))
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
