using LinearAlgebra, Statistics, Printf
using CairoMakie
using KernelAbstractions
const KA = KernelAbstractions

include("helpers.jl")

@kernel inbounds = true function init!(ϕ, k_ηf, ρt, Pt, Pf, pa_phys)
    (; xc, yc, ly, w, ϕ_bg, ϕ_A, k_ηf0, npow, ρf, ρs, ρt_ref) = pa_phys
    ix, iy = @index(Global, NTuple)
    ϕ[ix, iy]    = ϕ_bg * (1.0 + ϕ_A * exp(-(xc[ix] / w) ^ 2 - ((yc[iy] + ly / 4) / w) ^ 2))
    k_ηf[ix, iy] = k_ηf0 * (ϕ[ix, iy] / ϕ_bg) ^ npow
    ρt[ix, iy]   = (1.0 - ϕ[ix, iy]) * ρs + ϕ[ix, iy] * ρf
    Pt[ix, iy]   = -ρt_ref * yc[iy]
    Pf[ix, iy]   = Pt[ix, iy]
end

@kernel inbounds = true function copy_old!(ϕ_old, ρt_old, ϕ, pa_phys)
    (; ρf, ρs) = pa_phys
    ix, iy = @index(Global, NTuple)
    ϕ_old[ix, iy]  = ϕ[ix, iy]
    ρt_old[ix, iy] = (1.0 - ϕ[ix, iy]) * ρs + ϕ[ix, iy] * ρf
end

# Update material params, θ_dτ, and re_kf scratch (all nx×ny)
@kernel inbounds = true function update_params!(ηϕ, k_ηf, ρt, θ_dτ, re_kf, ϕ, Pf, Pt, pa_phys, ϕ_rel, lmin, cfl, maxdxy)
    (; ϕ_bg, ηϕ_bg, k_ηf0, npow, R, λPe, ρf, ρs) = pa_phys
    ix, iy = @index(Global, NTuple)
    Pe_ij         = Pf[ix, iy] - Pt[ix, iy]
    ηϕ[ix, iy]    = ηϕ_bg * (ϕ_bg / ϕ[ix, iy]) * (1.0 + 0.5 * (1.0 / R - 1.0) * (1.0 + tanh(Pe_ij / λPe)))
    k_ηf[ix, iy]  = exp((1.0 - ϕ_rel) * log(k_ηf[ix, iy]) + ϕ_rel * log(k_ηf0 * (ϕ[ix, iy] / ϕ_bg) ^ npow))
    ρt[ix, iy]    = (1.0 - ϕ[ix, iy]) * ρs + ϕ[ix, iy] * ρf
    lc_loc_ij     = sqrt(k_ηf[ix, iy] * ηϕ[ix, iy])
    re_ij         = π + sqrt(π ^ 2 + (lmin / lc_loc_ij) ^ 2)
    θ_dτ[ix, iy]  = lmin / re_ij / cfl / maxdxy
    re_kf[ix, iy] = re_ij * k_ηf[ix, iy]
end

# Compute dτ_βf on interior (nx-2)×(ny-2) using maxloc of re*k_ηf over 5-point stencil
@kernel inbounds = true function update_dτ_βf!(dτ_βf, re_kf, pa_phys, cfl, lmin, maxdxy)
    ix, iy = @index(Global, NTuple)
    # maxloc: max of cell and 4 face-adjacent neighbours (maps interior index to full array offset +1)
    re_kf_max = max(re_kf[ix+1, iy+1],
                max(re_kf[ix,   iy+1], re_kf[ix+2, iy+1]),
                max(re_kf[ix+1, iy  ], re_kf[ix+1, iy+2]))
    dτ_βf[ix, iy] = cfl * lmin * maxdxy / re_kf_max
end

# Solid velocity update (damped pseudo-transient)
@kernel inbounds = true function update_Vs!(Vs, R, τ, Pt, ρt, pa_phys, νdτ, _dx, _dy)
    (; ηs) = pa_phys
    ix, iy = @index(Global, NTuple)
    if @isin(R.Vx)
        R.Vx[ix, iy]    = -∂x(Pt, ix, iy+1, _dx) + ∂x(τ.xx, ix, iy+1, _dx) + ∂y(τ.xy, ix, iy, _dy)
        Vs.x[ix, iy+1] += R.Vx[ix, iy] * νdτ / ηs
    end
    if @isin(R.Vy)
        R.Vy[ix, iy]    = -∂y(Pt, ix+1, iy, _dy) + ∂y(τ.yy, ix+1, iy, _dy) + ∂x(τ.xy, ix, iy, _dx) - avy(ρt, ix+1, iy)
        Vs.y[ix+1, iy] += R.Vy[ix, iy] * νdτ / ηs
    end
end

# Stress, solid pressure update
@kernel inbounds = true function update_tau_Pt!(τ, R, Vs, ∇Vs, Pt, ϕ, ϕ_old, pa_phys, dτ_r, dτ_Pr, dt, _dx, _dy)
    (; ηs) = pa_phys
    ix, iy = @index(Global, NTuple)
    if @isin(∇Vs) ∇Vs[ix, iy] = ∂x(Vs.x, ix, iy+1, _dx) + ∂y(Vs.y, ix+1, iy, _dy) end
    if @isin(R.τxx)
        R.τxx[ix, iy]     = -τ.xx[ix+1, iy+1] + 2.0 * ηs * (∂x(Vs.x, ix, iy+1, _dx) - ∇Vs[ix, iy] / 3.0)
        τ.xx[ix+1, iy+1] += R.τxx[ix, iy] * dτ_r
    end
    if @isin(R.τyy)
        R.τyy[ix, iy]     = -τ.yy[ix+1, iy+1] + 2.0 * ηs * (∂y(Vs.y, ix+1, iy, _dy) - ∇Vs[ix, iy] / 3.0)
        τ.yy[ix+1, iy+1] += R.τyy[ix, iy] * dτ_r
    end
    if @isin(τ.xy)
        τ.xy[ix, iy] += dτ_r * (-τ.xy[ix, iy] + ηs * (∂y(Vs.x, ix, iy, _dy) + ∂x(Vs.y, ix, iy, _dx)))
    end
    if @isin(R.Pt)
        R.Pt[ix, iy]    = -(ϕ[ix+1, iy+1] - ϕ_old[ix+1, iy+1]) / dt +
                          ∂x(ϕ, Vs.x, ix, iy+1, _dx) + ∂y(ϕ, Vs.y, ix+1, iy, _dy)
        Pt[ix+1, iy+1] -= R.Pt[ix, iy] * ηs * (1.0 - ϕ[ix+1, iy+1]) * dτ_Pr
    end
end

# Darcy flux update (damped)
@kernel inbounds = true function update_qD!(qD, Qf, Pf, k_ηf, Vs, ρt, θ_dτ, pa_phys, _dx, _dy)
    (; ρf) = pa_phys
    ix, iy = @index(Global, NTuple)
    if @isin(qD.x)
        Rq = -qD.x[ix, iy] - avx(k_ηf, ix, iy+1) * ∂x(Pf, ix, iy+1, _dx)
        qD.x[ix, iy] += Rq / (1.0 + avx(θ_dτ, ix, iy+1))
    end
    if @isin(qD.y)
        Rq = -qD.y[ix, iy] - avy(k_ηf, ix+1, iy) * (∂y(Pf, ix+1, iy, _dy) - ρf)
        qD.y[ix, iy] += Rq / (1.0 + avy(θ_dτ, ix+1, iy))
    end
    if @isin(Qf.x) Qf.x[ix, iy] = ρf * qD.x[ix, iy] + avx(ρt, ix, iy+1) * Vs.x[ix, iy+1] end
    if @isin(Qf.y) Qf.y[ix, iy] = ρf * qD.y[ix, iy] + avy(ρt, ix+1, iy) * Vs.y[ix+1, iy] end
end

# Fluid pressure from total mass balance + porosity from compaction equation
@kernel inbounds = true function update_Pf_phi!(Pf, ϕ, ϕ_old, Pt, ηϕ, ρt, ρt_old, Qf, R, dτ_βf, dt, _dx, _dy)
    ix, iy = @index(Global, NTuple)
    if @isin(R.Pf)
        R.Pf[ix, iy]    = (ρt[ix+1, iy+1] - ρt_old[ix+1, iy+1]) / dt +
                          ∂x(Qf.x, ix, iy, _dx) + ∂y(Qf.y, ix, iy, _dy)
        Pf[ix+1, iy+1] -= R.Pf[ix, iy] * ϕ[ix+1, iy+1] * dτ_βf[ix, iy]
        dϕdt            = (Pf[ix+1, iy+1] - Pt[ix+1, iy+1]) / ηϕ[ix+1, iy+1]
        ϕ[ix+1, iy+1]   = ϕ_old[ix+1, iy+1] + dt * dϕdt
    end
end

@views function main(backend=CPU(); dtype=Float64)
    # independent physics
    lc     = 1.0
    Δρg    = 1.0
    ηϕ_bg  = 1.0
    # scales
    psc    = Δρg * lc
    tsc    = ηϕ_bg / psc
    # nondim numbers
    R      = 100.0
    ly_lc  = 80.0 / sqrt(0.9R)
    w_lc   = 5.0 / sqrt(0.9R)
    ρs_ρf  = 1.2
    ϕ_bg   = 0.005
    ϕ_A    = 3.0
    npow   = 3.0
    ra     = 2.0
    # dependent physics
    λPe    = 0.01 * psc
    ηs     = ηϕ_bg * ϕ_bg
    ly     = ly_lc * lc
    lx     = ly / ra
    w      = w_lc * lc
    k_ηf0  = lc^2 / ηϕ_bg
    dt     = 1e-6 * tsc
    # numerics
    nx      = 40
    ny      = ceil(Int, nx * ra)
    tol     = 1e-3
    cfl_dt  = 2e-2
    ϕ_rel   = 1e-1
    nt      = 1 #150
    miniter = 10
    maxiter = 150ny
    nout    = 5ny
    nviz    = 5
    top     = 0.9ny
    # preprocessing
    dx, dy  = lx / nx, ly / ny
    xc      = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc      = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    ρf      = Δρg / (ρs_ρf - 1.0)
    ρs      = ρf * ρs_ρf
    # dmp solid (global scalars, same as _dmp)
    re_m    = 2.3π
    r       = 0.5
    lτ_re_m = min(lx, ly) / re_m
    vdτ     = min(dx, dy) / sqrt(4.1)
    θ_dτ_s  = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ_s + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ_s
    # dmp fluid (local arrays)
    cfl     = 1 / sqrt(2.1)
    lmin    = min(lx, ly)
    maxdxy  = max(dx, dy)
    _dx, _dy = 1.0 / dx, 1.0 / dy
    # physics named tuple
    ρt_ref  = (1.0 - ϕ_bg) * ρs + ϕ_bg * ρf
    pa_phys = (; xc, yc, ly, w, ϕ_bg, ηϕ_bg, ϕ_A, k_ηf0, npow, R, λPe, ρf, ρs, ρt_ref, ηs)
    # arrays
    ϕ        = KA.zeros(backend, dtype, nx, ny)
    k_ηf     = KA.zeros(backend, dtype, nx, ny)
    ηϕ       = KA.zeros(backend, dtype, nx, ny)
    Pt       = KA.zeros(backend, dtype, nx, ny)
    Pf       = KA.zeros(backend, dtype, nx, ny)
    ρt       = KA.zeros(backend, dtype, nx, ny)
    ϕ_old    = KA.zeros(backend, dtype, nx, ny)
    ρt_old   = KA.zeros(backend, dtype, nx, ny)
    θ_dτ     = KA.zeros(backend, dtype, nx, ny)
    re_kf    = KA.zeros(backend, dtype, nx, ny)
    dτ_βf    = KA.zeros(backend, dtype, nx - 2, ny - 2)
    ∇Vs      = KA.zeros(backend, dtype, nx - 2, ny - 2)
    Vs = (x=KA.zeros(backend, dtype, nx - 1, ny),
          y=KA.zeros(backend, dtype, nx, ny - 1),)
    τ = (xx=KA.zeros(backend, dtype, nx, ny),
         yy=KA.zeros(backend, dtype, nx, ny),
         xy=KA.zeros(backend, dtype, nx - 1, ny - 1),)
    qD = (x=KA.zeros(backend, dtype, nx - 1, ny - 2),
          y=KA.zeros(backend, dtype, nx - 2, ny - 1),)
    Qf = (x=KA.zeros(backend, dtype, nx - 1, ny - 2),
          y=KA.zeros(backend, dtype, nx - 2, ny - 1),)
    R = (Vx =KA.zeros(backend, dtype, nx - 1, ny - 2),
         Vy =KA.zeros(backend, dtype, nx - 2, ny - 1),
         Pt =KA.zeros(backend, dtype, nx - 2, ny - 2),
         Pf =KA.zeros(backend, dtype, nx - 2, ny - 2),
         τxx=KA.zeros(backend, dtype, nx - 2, ny - 2),
         τyy=KA.zeros(backend, dtype, nx - 2, ny - 2),)
    # init
    init!(backend, 256, (nx, ny))(ϕ, k_ηf, ρt, Pt, Pf, pa_phys)
    KA.synchronize(backend)
    # visualisation init
    fig = Figure(; size=(800, 600))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Porosity"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="P effective"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="permeability"),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="bulk viscosity"))
    plt = (heatmap!(axs[1], xc, yc, Array(ϕ);        colormap=:turbo),
           heatmap!(axs[2], xc, yc, Array(Pf .- Pt);  colormap=:turbo),
           heatmap!(axs[3], xc, yc, Array(k_ηf);      colormap=:turbo),
           heatmap!(axs[4], xc, yc, Array(ηϕ);        colormap=:turbo))
    cbs = (Colorbar(fig[1, 1][1, 2], plt[1]),
           Colorbar(fig[1, 2][1, 2], plt[2]),
           Colorbar(fig[2, 1][1, 2], plt[3]),
           Colorbar(fig[2, 2][1, 2], plt[4]))
    hideydecorations!.((axs[2], axs[4]))
    display(fig)
    nxm = (nx + 1) ÷ 2
    time_evo = [0.0]
    ϕmax_evo = [maximum(ϕ)]
    # time loop
    for it = 1:nt
        println("> step $it")
        copy_old!(backend, 256, (nx, ny))(ϕ_old, ρt_old, ϕ, pa_phys)
        iter  = 0
        error = 10tol
        # pseudo-transient loop
        while (error > tol && iter <= maxiter) || iter < miniter
            iter += 1
            update_params!(backend, 256, (nx, ny))(ηϕ, k_ηf, ρt, θ_dτ, re_kf, ϕ, Pf, Pt, pa_phys, ϕ_rel, lmin, cfl, maxdxy)
            update_dτ_βf!(backend, 256, (nx - 2, ny - 2))(dτ_βf, re_kf, pa_phys, cfl, lmin, maxdxy)
            update_Vs!(backend, 256, (nx, ny))(Vs, R, τ, Pt, ρt, pa_phys, νdτ, _dx, _dy)
            update_tau_Pt!(backend, 256, (nx, ny))(τ, R, Vs, ∇Vs, Pt, ϕ, ϕ_old, pa_phys, dτ_r, dτ_Pr, dt, _dx, _dy)
            update_qD!(backend, 256, (nx, ny))(qD, Qf, Pf, k_ηf, Vs, ρt, θ_dτ, pa_phys, _dx, _dy)
            update_Pf_phi!(backend, 256, (nx, ny))(Pf, ϕ, ϕ_old, Pt, ηϕ, ρt, ρt_old, Qf, R, dτ_βf, dt, _dx, _dy)
            neumann_bcs!(ϕ)
            # error check
            if mod(iter, nout) == 0
                KA.synchronize(backend)
                errs = [maximum(abs, R.Vx), maximum(abs, R.Vy), maximum(abs, R.Pt), maximum(abs, R.Pf)]
                error = maximum(errs)
                @printf "  iter/ny=%d, errs: RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e \n" ceil(iter/ny) errs...
            end
        end
        KA.synchronize(backend)
        push!(time_evo, time_evo[end] + dt)
        push!(ϕmax_evo, maximum(ϕ))
        # time step from compaction rate
        dϕdt_max = maximum(abs, inn((Pf .- Pt) ./ ηϕ))
        dt = cfl_dt * maximum(ϕ) / dϕdt_max
        # visualisation
        if mod(it, nviz) == 0 || it == 1
            plt[1][3] = Array(ϕ)
            plt[2][3] = Array(Pf .- Pt)
            plt[3][3] = Array(k_ηf)
            plt[4][3] = Array(ηϕ)
            display(fig)
        end
        # stopping criterion
        ((inv(ϕ_bg) * Array(ϕ)[nxm, Int(ceil(top))]) > 1.05) && break
    end
end

main() # CPU
# main(CUDABackend()) # Nvidia GPU
# main(ROCBackend()) # AMD GPU
