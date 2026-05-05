using LinearAlgebra, Statistics, Printf
using CairoMakie
using Reactant

const _CUDA_functional = try
    using CUDA
    CUDA.functional()
catch
    false
end

# backend init
function init_backend(backend::Symbol=:auto)
    # backend: :auto (detect CUDA), :gpu, :cpu, :none (plain Julia, no Reactant)
    resolved = if backend === :auto
        _CUDA_functional ? :gpu : :cpu
    else
        backend
    end
    if resolved !== :none
        Reactant.set_default_backend(resolved === :gpu ? "gpu" : "cpu")
        @info "Reactant backend: $resolved"
        return resolved, Reactant.ConcreteRArray, Reactant.ConcreteRNumber
    else
        @info "Plain Julia backend"
        return resolved, identity, identity
    end
end

# helper functions
to_scalar(x::Union{AbstractFloat, Integer}) = x
to_scalar(x) = Reactant.to_number(x)
@views inn(A)    = A[2:end-1, 2:end-1]
@views avx(A)    = @. 0.5 * (A[1:end-1, :] + A[2:end, :])
@views avy(A)    = @. 0.5 * (A[:, 1:end-1] + A[:, 2:end])
@views avxi(A)   = @. 0.5 * (A[1:end-1, 2:end-1] + A[2:end, 2:end-1])
@views avyi(A)   = @. 0.5 * (A[2:end-1, 1:end-1] + A[2:end-1, 2:end])
@views av(A)     = @. 0.25 * (A[1:end-1, 1:end-1] + A[1:end-1, 2:end] + A[2:end, 1:end-1] + A[2:end, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                   max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))
@views function neumann_bcs_ap!(A)
    A[1, :]   .= A[2, :]
    A[end, :] .= A[end-1, :]
    A[:, 1]   .= A[:, 2]
    A[:, end] .= A[:, end-1]
end

@views function step(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                     Vxs, Vys, τxx, τyy, τxy, ∇Vs, qDx, qDy, Qfx, Qfy, dϕdt,
                     lc_loc, re, θ_dτ, dτ_βf,
                     RVx, RVy, RPt, RPf, Rτxx, Rτyy, Rτxy, RqDx, RqDy,
                     ϕ_old, ρt_old,
                     dx, dy, dt, nout,
                     ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                     νdτ, dτ_r, dτ_Pr, cfl, lx, ly,
                     iter, err, err_log)
    iter += 1
    # material params
    ηϕ   .= ηϕ_bg .* (ϕ_bg ./ ϕ) .* (1.0 .+ 0.5 .* (1.0 / R - 1.0) .* (1.0 .+ tanh.(Pe ./ λPe)))
    k_ηf .= exp.((1.0 - ϕ_rel) .* log.(k_ηf) .+ ϕ_rel .* log.(k_ηf0 .* (ϕ ./ ϕ_bg) .^ npow))
    ρt   .= (1.0 .- ϕ) .* ρs .+ ϕ .* ρf
    ∇Vs  .= diff(Vxs[:, 2:end-1], dims=1) ./ dx .+ diff(Vys[2:end-1, :], dims=2) ./ dy
    # solid velocities
    RVx  .= diff(.-Pt[:, 2:end-1] .+ τxx[:, 2:end-1], dims=1) ./ dx .+ diff(τxy, dims=2) ./ dy
    RVy  .= diff(.-Pt[2:end-1, :] .+ τyy[2:end-1, :], dims=2) ./ dy .+ diff(τxy, dims=1) ./ dx .- avy(ρt[2:end-1, :])
    Vxs[:, 2:end-1] .+= RVx .* νdτ ./ ηs
    Vys[2:end-1, :] .+= RVy .* νdτ ./ ηs
    # linear viscous flow law
    Rτxx[2:end-1, 2:end-1] .= -inn(τxx) .+ 2.0 .* ηs .* (diff(Vxs[:, 2:end-1], dims=1) ./ dx .- ∇Vs ./ 3)
    Rτyy[2:end-1, 2:end-1] .= -inn(τyy) .+ 2.0 .* ηs .* (diff(Vys[2:end-1, :], dims=2) ./ dy .- ∇Vs ./ 3)
    Rτxy                    .=    -τxy   .+        ηs .* (diff(Vxs, dims=2) ./ dy .+ diff(Vys, dims=1) ./ dx)
    τxx .+= Rτxx .* dτ_r
    τyy .+= Rτyy .* dτ_r
    τxy .+= Rτxy .* dτ_r
    # total pressure from solid mass balance
    RPt .= .-inn(ϕ .- ϕ_old) ./ dt .+
           diff((1.0 .- avxi(ϕ)) .* Vxs[:, 2:end-1], dims=1) ./ dx .+
           diff((1.0 .- avyi(ϕ)) .* Vys[2:end-1, :], dims=2) ./ dy
    # porous flow
    lc_loc .= sqrt.(k_ηf .* ηϕ)
    re     .= π .+ sqrt.(π^2 .+ (min(lx, ly) ./ lc_loc) .^ 2)
    θ_dτ   .= min(lx, ly) ./ re ./ cfl ./ max(dx, dy)
    dτ_βf  .= cfl * min(lx, ly) .* max(dx, dy) ./ maxloc(re .* k_ηf)
    RqDx   .= .-qDx .- avxi(k_ηf) .*  diff(Pf[:, 2:end-1], dims=1) ./ dx
    RqDy   .= .-qDy .- avyi(k_ηf) .* (diff(Pf[2:end-1, :], dims=2) ./ dy .- ρf)
    qDx   .+= RqDx ./ (1.0 .+ avxi(θ_dτ))
    qDy   .+= RqDy ./ (1.0 .+ avyi(θ_dτ))
    Qfx    .= ρf .* qDx .+ avxi(ρt) .* Vxs[:, 2:end-1]
    Qfy    .= ρf .* qDy .+ avyi(ρt) .* Vys[2:end-1, :]
    Pt[2:end-1, 2:end-1] .-= RPt .* ηs .* (1.0 .- inn(ϕ)) .* dτ_Pr
    # fluid pressure from total mass balance
    RPf .= inn(ρt .- ρt_old) ./ dt .+ diff(Qfx, dims=1) ./ dx .+ diff(Qfy, dims=2) ./ dy
    Pf[2:end-1, 2:end-1] .-= RPf .* inn(ϕ) .* dτ_βf
    # porosity update from compaction eqn
    Pe   .= Pf .- Pt
    dϕdt .= inn(Pe ./ ηϕ)
    ϕ[2:end-1, 2:end-1] .= inn(ϕ_old) .+ dt .* dϕdt
    neumann_bcs_ap!(ϕ)
    # err (ifelse to avoid branching on traced bool)
    err  = ifelse(mod(iter, nout) == 0,
                  max(maximum(abs, RVx), maximum(abs, RVy), maximum(abs, RPt), maximum(abs, RPf)),
                  err)
    mask    = (1:length(err_log)) .== ((iter - 1) ÷ nout + 1)
    err_log .= ifelse.(mask, err, err_log)
    return iter, err
end

function solve(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
               Vxs, Vys, τxx, τyy, τxy, ∇Vs, qDx, qDy, Qfx, Qfy, dϕdt,
               lc_loc, re, θ_dτ, dτ_βf,
               RVx, RVy, RPt, RPf, Rτxx, Rτyy, Rτxy, RqDx, RqDy,
               ϕ_old, ρt_old,
               dx, dy, dt, nout,
               ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
               νdτ, dτ_r, dτ_Pr, cfl, lx, ly,
               tol, maxiter, iter, err, err_log)
    ϕ_old  .= ϕ
    ρt_old .= (1.0 .- ϕ) .* ρs .+ ϕ .* ρf
    @trace while (iter < maxiter) & (err >= tol)
        iter, err = step(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                         Vxs, Vys, τxx, τyy, τxy, ∇Vs, qDx, qDy, Qfx, Qfy, dϕdt,
                         lc_loc, re, θ_dτ, dτ_βf,
                         RVx, RVy, RPt, RPf, Rτxx, Rτyy, Rτxy, RqDx, RqDy,
                         ϕ_old, ρt_old,
                         dx, dy, dt, nout,
                         ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                         νdτ, dτ_r, dτ_Pr, cfl, lx, ly,
                         iter, err, err_log)
    end
    return iter, err
end

function main(; nx=40, backend=:auto, verbose=true)
    resolved, CRArray, CRNumber = init_backend(backend)
    use_reactant = resolved !== :none
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
    dt0    = 1e-6 * tsc
    # numerics
    ny      = ceil(Int, nx * ra)
    tol     = 1e-3
    cfl_dt  = 2e-2
    ϕ_rel   = 1e-1
    nt      = 1
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
    # dmp solid
    re_m    = 2.3π
    r       = 0.5
    lτ_re_m = min(lx, ly) / re_m
    vdτ     = min(dx, dy) / sqrt(4.1)
    θ_dτ_s  = lτ_re_m * (r + 4 / 3) / vdτ   # scalar (name freed below for array)
    dτ_r    = 1.0 / (θ_dτ_s + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ_s
    cfl     = 1 / sqrt(2.1)
    # initial conditions
    ϕ       = CRArray(ϕ_bg .* (1 .+ ϕ_A .* exp.(-(xc ./ w) .^ 2 .- ((yc' .+ ly / 4) ./ w) .^ 2)))
    k_ηf    = CRArray(k_ηf0 .* (Array(ϕ) ./ ϕ_bg) .^ npow)
    ρt      = CRArray((1.0 .- Array(ϕ)) .* ρs .+ Array(ϕ) .* ρf)
    ρt_ref  = (1.0 .- ϕ_bg) .* ρs .+ ϕ_bg .* ρf
    Pt      = CRArray(-ρt_ref .* (0.0 .* xc .+ yc'))
    Pf      = CRArray(Array(Pt))
    Pe      = CRArray(zeros(nx, ny))
    ϕ_old   = CRArray(Array(ϕ))
    ρt_old  = CRArray(Array(ρt))
    # initialisation
    ηϕ      = CRArray(zeros(nx, ny))
    Vxs     = CRArray(zeros(nx - 1, ny))
    Vys     = CRArray(zeros(nx, ny - 1))
    τxx     = CRArray(zeros(nx, ny))
    τyy     = CRArray(zeros(nx, ny))
    τxy     = CRArray(zeros(nx - 1, ny - 1))
    ∇Vs     = CRArray(zeros(nx - 2, ny - 2))
    dϕdt    = CRArray(zeros(nx - 2, ny - 2))
    qDx     = CRArray(zeros(nx - 1, ny - 2))
    qDy     = CRArray(zeros(nx - 2, ny - 1))
    Qfx     = CRArray(zeros(nx - 1, ny - 2))
    Qfy     = CRArray(zeros(nx - 2, ny - 1))
    RVx     = CRArray(zeros(nx - 1, ny - 2))
    RVy     = CRArray(zeros(nx - 2, ny - 1))
    RPt     = CRArray(zeros(nx - 2, ny - 2))
    RPf     = CRArray(zeros(nx - 2, ny - 2))
    Rτxx    = CRArray(zeros(nx, ny))
    Rτyy    = CRArray(zeros(nx, ny))
    Rτxy    = CRArray(zeros(nx - 1, ny - 1))
    RqDx    = CRArray(zeros(nx - 1, ny - 2))
    RqDy    = CRArray(zeros(nx - 2, ny - 1))
    lc_loc  = CRArray(zeros(nx, ny))
    re      = CRArray(zeros(nx, ny))
    θ_dτ    = CRArray(zeros(nx, ny))
    dτ_βf   = CRArray(zeros(nx - 2, ny - 2))
    dt_r    = CRNumber(dt0)
    iter    = CRNumber(0)
    err     = CRNumber(10tol)
    err_log = CRArray(zeros(maxiter ÷ nout))
    # visualisation init
    fig = Figure(; size=(600, 800))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Porosity"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="P effective"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="permeability"),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="bulk viscosity"))
    plt = (heatmap!(axs[1], xc, yc, Array(ϕ);        colormap=:turbo),
           heatmap!(axs[2], xc, yc, Array(Pf .- Pt); colormap=:turbo),
           heatmap!(axs[3], xc, yc, Array(k_ηf);     colormap=:turbo),
           heatmap!(axs[4], xc, yc, Array(ηϕ);       colormap=:turbo))
    cbs = (Colorbar(fig[1, 1][1, 2], plt[1]),
           Colorbar(fig[1, 2][1, 2], plt[2]),
           Colorbar(fig[2, 1][1, 2], plt[3]),
           Colorbar(fig[2, 2][1, 2], plt[4]))
    hideydecorations!.((axs[2], axs[4]))

    nxm = (nx + 1) ÷ 2
    time_evo = [0.0]
    ϕmax_evo = [to_scalar(maximum(ϕ))]

    # compile once before time loop
    if use_reactant
        t_compile = time_ns()
        solve_react = @compile sync=true solve(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                                               Vxs, Vys, τxx, τyy, τxy, ∇Vs, qDx, qDy, Qfx, Qfy, dϕdt,
                                               lc_loc, re, θ_dτ, dτ_βf,
                                               RVx, RVy, RPt, RPf, Rτxx, Rτyy, Rτxy, RqDx, RqDy,
                                               ϕ_old, ρt_old,
                                               dx, dy, dt_r, nout,
                                               ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                                               νdτ, dτ_r, dτ_Pr, cfl, lx, ly,
                                               tol, maxiter, iter, err, err_log)
        t_compile = (time_ns() - t_compile) / 1e9
        @printf "  compiled in %.3f s\n" t_compile
    end

    # time loop
    for it = 1:nt
        println("> step $it")
        iter    = CRNumber(0)
        err     = CRNumber(10tol)
        err_log = CRArray(zeros(maxiter ÷ nout))

        if use_reactant
            t_run = time_ns()
            iter, err = solve_react(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                                    Vxs, Vys, τxx, τyy, τxy, ∇Vs, qDx, qDy, Qfx, Qfy, dϕdt,
                                    lc_loc, re, θ_dτ, dτ_βf,
                                    RVx, RVy, RPt, RPf, Rτxx, Rτyy, Rτxy, RqDx, RqDy,
                                    ϕ_old, ρt_old,
                                    dx, dy, dt_r, nout,
                                    ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                                    νdτ, dτ_r, dτ_Pr, cfl, lx, ly,
                                    tol, maxiter, iter, err, err_log)
            t_run = (time_ns() - t_run) / 1e9
            @printf "  run: %.3f s\n" t_run
        else
            t_run = time_ns()
            iter, err = solve(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                              Vxs, Vys, τxx, τyy, τxy, ∇Vs, qDx, qDy, Qfx, Qfy, dϕdt,
                              lc_loc, re, θ_dτ, dτ_βf,
                              RVx, RVy, RPt, RPf, Rτxx, Rτyy, Rτxy, RqDx, RqDy,
                              ϕ_old, ρt_old,
                              dx, dy, dt_r, nout,
                              ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                              νdτ, dτ_r, dτ_Pr, cfl, lx, ly,
                              tol, maxiter, iter, err, err_log)
            t_run = (time_ns() - t_run) / 1e9
            @printf "  run: %.3f s\n" t_run
        end
        @printf "  converged: iter/ny=%d, err=%1.3e\n" to_scalar(iter) ÷ ny to_scalar(err)
        if verbose
            for (i, e) in enumerate(Array(err_log))
                e == 0 && break  # stop at first unfilled slot
                @printf "  iter/ny=%d, errs=%1.3e\n" i * nout ÷ ny e
            end
        end

        # diagnostic monitoring
        push!(time_evo, time_evo[end] + to_scalar(dt_r))
        push!(ϕmax_evo, to_scalar(maximum(ϕ)))
        # time step update (extract to host for arithmetic)
        dt_r = CRNumber(cfl_dt * to_scalar(maximum(ϕ)) / maximum(abs, Array(dϕdt)))

        # visualisation
        if mod(it, nviz) == 0 || it == 1
            plt[1][3] = Array(ϕ)
            plt[2][3] = Array(Pf .- Pt)
            plt[3][3] = Array(k_ηf)
            plt[4][3] = Array(ηϕ)
            display(fig)
        end
        # stopping criterion
        ((inv(ϕ_bg) * to_scalar(ϕ[nxm, Int(ceil(top))])) > 1.05) && break
    end
    return
end

main(nx=64, backend=:none, verbose=false)
