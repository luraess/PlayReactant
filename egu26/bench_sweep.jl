# Resolution sweep for all 4 Reactant physics solvers.
# Run from the repo root: julia --project scripts/bench_sweep.jl
#
# Each script is wrapped in its own module to avoid namespace clashes.
# The `_bench_sweep` flag suppresses the hardcoded main() call at the
# bottom of each included script.
using Printf

const _bench_sweep = true   # must be in Main before any include

println("Loading solvers (this compiles Julia code, not Reactant)...")
module S2D;  include("Stokes_react2D.jl");           end
module S3D;  include("Stokes_react3D.jl");           end
module PW2D; include("PorowavesStokes_react2D.jl");  end
module PW3D; include("PorowavesStokes_react3D.jl");  end

backend = :gpu   # change to :cpu to run on CPU

# ─────────────────────────────────────────────────────────
# Stokes 2D  (14 arrays per iteration, ~nx × ny each)
# ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Stokes 2D")
println("="^60)
results_s2d = []
# for nx in [64, 128]
for nx in [64, 128, 256, 512, 1024, 2048]
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
# for nx in [32, 64]
for nx in [32, 64, 128, 256, 512]
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
# for nx in [32, 64]
for nx in [32, 64, 128, 256, 512, 1024]
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
# for nx in [32, 64]
for nx in [16, 32, 64, 128, 256]
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
for (label, results) in [("Stokes2D",   results_s2d),
                         ("Stokes3D",   results_s3d),
                         ("PorWaves2D", results_pw2d),
                         ("PorWaves3D", results_pw3d)]
    for r in results
        @printf "%-20s %6d %12.2f %10.3f %8d %10.2f\n" label r.nx r.t_compile r.t_run r.niter r.T_eff
    end
end
