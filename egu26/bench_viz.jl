# Standalone visualisation for bench_sweep results.
# Usage:
#   julia --project egu26/bench_viz.jl                          # uses bench_sweep_results.toml next to this file
#   BENCH_SWEEP_OUT=/path/to/results.toml julia --project egu26/bench_viz.jl
using CairoMakie, TOML

out_file = get(ENV, "BENCH_SWEEP_OUT", joinpath(@__DIR__, "bench_sweep_results.toml"))
isfile(out_file) || error("Results file not found: $out_file\nSet BENCH_SWEEP_OUT to point at your .toml")

toml_data = TOML.parsefile(out_file)
runs      = toml_data["runs"]

solvers = unique(r["solver"] for r in runs)
colors  = Makie.wong_colors()
markers = [:circle, :rect, :diamond, :utriangle]
solver_colors  = Dict(s => colors[i]  for (i, s) in enumerate(solvers))
solver_markers = Dict(s => markers[i] for (i, s) in enumerate(solvers))

fig = Figure(; size=(900, 700))
ax_trun  = Axis(fig[1, 1]; xlabel="nx",     ylabel="t_run [s]",         title="Wall time",
                xscale=log2, yscale=log10)
ax_teff  = Axis(fig[1, 2]; xlabel="nx",     ylabel="T_eff [GB/s]",      title="Effective memory throughput",
                xscale=log2)
ax_niter = Axis(fig[2, 1]; xlabel="nx",     ylabel="niter",             title="Iterations to convergence",
                xscale=log2)
ax_comp  = Axis(fig[2, 2]; xlabel="solver", ylabel="min t_compile [s]", title="Min compile time")

for s in solvers
    sr    = filter(r -> r["solver"] == s, runs)
    idxs  = sortperm([r["nx"] for r in sr])
    sr    = sr[idxs]
    nxs    = [r["nx"]             for r in sr]
    t_runs = [r["t_run"]          for r in sr]
    teffs  = [r["T_eff"]          for r in sr]
    niters = [Float64(r["niter"]) for r in sr]
    c, mk  = solver_colors[s], solver_markers[s]
    scatterlines!(ax_trun,  nxs, t_runs; color=c, marker=mk, label=s)
    scatterlines!(ax_teff,  nxs, teffs;  color=c, marker=mk, label=s)
    scatterlines!(ax_niter, nxs, niters; color=c, marker=mk, label=s)
end
Legend(fig[1, 3], ax_trun; framevisible=false)

compile_mins = [minimum(r["t_compile"] for r in runs if r["solver"] == s) for s in solvers]
barplot!(ax_comp, 1:length(solvers), compile_mins;
         color=[solver_colors[s] for s in solvers])
ax_comp.xticks = (1:length(solvers), solvers)

viz_file = replace(out_file, r"\.toml$" => ".png")
save(viz_file, fig)
println("Figure saved to: $viz_file")
