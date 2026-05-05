using CairoMakie

# -----------------------------------------------------------------------
# Benchmark data extracted from run output files
# local tile: 8192×8192 (Float64), nt=10, Ndev=N, mean of 5 runs
# -----------------------------------------------------------------------

ngpus = [4, 16, 64, 256]

# Run 5fqe  (2026-04-29T18-35-00.290_5fqe)
Teff_5fqe = [2130.2, 1195.23, 121.65, 28.77]   # GB/s
t_5fqe    = [0.015122348200000002, 0.026951299800000002, 0.26480159400000003, 1.1196599452]  # s

# Run 1iIx  (2026-04-29T19-07-08.880_1iIx)
Teff_1iIx = [2136.67, 2099.84, 2039.85, 124.4]  # GB/s
t_1iIx    = [0.015076494800000001, 0.015340641000000002, 0.0157916536, 0.25894568160000003] # s

# Run xq3J  (2026-04-29T19-08-49.662_xq3J)
Teff_xq3J = [2235.07, 2205.36, 2079.41, 30.47]  # GB/s
t_xq3J    = [0.0144127742, 0.014606623400000001, 0.015491212600000002, 1.0571751632]       # s

# Weak scaling efficiency  (normalised to 4-GPU run of each series)
eff_5fqe = Teff_5fqe ./ Teff_5fqe[1] .* 100
eff_1iIx = Teff_1iIx ./ Teff_1iIx[1] .* 100
eff_xq3J = Teff_xq3J ./ Teff_xq3J[1] .* 100

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig = Figure(size=(900, 420), fontsize=14)

# Colours and markers
colors  = Makie.wong_colors()
c5fqe   = colors[1]
c1iIx   = colors[2]
cxq3J   = colors[3]
cideal  = (:black, 0.35)

ms = 10   # marker size
lw = 2    # line width

# ---- Panel 1: Teff ----
ax1 = Axis(fig[1, 1];
    xscale  = log2,
    xlabel  = "Number of GPUs",
    ylabel  = "T_eff  [GB/s]",
    title   = "Effective memory throughput (weak scaling)",
    xticks  = (ngpus, string.(ngpus)),
    xminorticksvisible = false,
)

# ideal (flat) reference from xq3J@4 GPUs as upper bound
hlines!(ax1, [Teff_xq3J[1]]; color=cideal, linestyle=:dash, linewidth=1.5, label="ideal (const.)")

lines!(ax1, ngpus, Teff_5fqe; color=c5fqe, linewidth=lw, label="5fqe")
scatter!(ax1, ngpus, Teff_5fqe; color=c5fqe, markersize=ms, marker=:circle)

lines!(ax1, ngpus, Teff_1iIx; color=c1iIx, linewidth=lw, label="1iIx")
scatter!(ax1, ngpus, Teff_1iIx; color=c1iIx, markersize=ms, marker=:utriangle)

lines!(ax1, ngpus, Teff_xq3J; color=cxq3J, linewidth=lw, label="xq3J")
scatter!(ax1, ngpus, Teff_xq3J; color=cxq3J, markersize=ms, marker=:rect)

axislegend(ax1; position=:lb, framevisible=true)

# ---- Panel 2: Weak scaling efficiency ----
ax2 = Axis(fig[1, 2];
    xscale  = log2,
    xlabel  = "Number of GPUs",
    ylabel  = "Weak scaling efficiency  [%]",
    title   = "Weak scaling efficiency",
    xticks  = (ngpus, string.(ngpus)),
    yticks  = 0:20:100,
    yminorticksvisible = false,
    xminorticksvisible = false,
)
ylims!(ax2, 0, 110)

# ideal line
hlines!(ax2, [100.0]; color=cideal, linestyle=:dash, linewidth=1.5, label="ideal")

lines!(ax2, ngpus, eff_5fqe; color=c5fqe, linewidth=lw, label="5fqe")
scatter!(ax2, ngpus, eff_5fqe; color=c5fqe, markersize=ms, marker=:circle)

lines!(ax2, ngpus, eff_1iIx; color=c1iIx, linewidth=lw, label="1iIx")
scatter!(ax2, ngpus, eff_1iIx; color=c1iIx, markersize=ms, marker=:utriangle)

lines!(ax2, ngpus, eff_xq3J; color=cxq3J, linewidth=lw, label="xq3J")
scatter!(ax2, ngpus, eff_xq3J; color=cxq3J, markersize=ms, marker=:rect)

axislegend(ax2; position=:lb, framevisible=true)

# -----------------------------------------------------------------------
outfile = joinpath(@__DIR__, "output", "weak_scaling.png")
save(outfile, fig; px_per_unit=2)
println("Saved → $outfile")
