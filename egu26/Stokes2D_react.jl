using Printf
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
@views avx(A) = @. 0.5 * (A[1:end-1, :] + A[2:end, :])
@views avy(A) = @. 0.5 * (A[:, 1:end-1] + A[:, 2:end])
@views av(A)  = @. 0.25 * (A[1:end-1, 1:end-1] + A[1:end-1, 2:end] + A[2:end, 1:end-1] + A[2:end, 2:end])

@views function step(Pt, Vxs, Vys, τxx, τyy, τxy, ∇Vs, RVx, RVy, Rτxx, Rτyy, Rτxy, ηs, ηs_v,
                     dτ_r, νdτ, dτ_Pr, dx, dy, nout, iter, err, err_log)
    iter += 1
    ∇Vs  .= diff(Vxs[:, 2:end-1], dims=1) ./ dx .+ diff(Vys[2:end-1, :], dims=2) ./ dy
    Pt  .-= ∇Vs .* ηs .* dτ_Pr
    # solid velocities
    RVx  .= diff(.-Pt .+ τxx, dims=1) ./ dx .+ diff(τxy[2:end-1,:], dims=2) ./ dy
    RVy  .= diff(.-Pt .+ τyy, dims=2) ./ dy .+ diff(τxy[:,2:end-1], dims=1) ./ dx
    Vxs[2:end-1, 2:end-1] .+= RVx .* νdτ ./ avx(ηs)
    Vys[2:end-1, 2:end-1] .+= RVy .* νdτ ./ avy(ηs)
    # BC free slip
    Vxs[:,[1,end]] .= Vxs[:,[2,end-1]]
    Vys[[1,end],:] .= Vys[[2,end-1],:]
    # linear viscous flow law
    Rτxx .= .-τxx .+ 2.0 .* ηs .* (diff(Vxs[:, 2:end-1], dims=1) ./ dx .- ∇Vs ./ 3)
    Rτyy .= .-τyy .+ 2.0 .* ηs .* (diff(Vys[2:end-1, :], dims=2) ./ dy .- ∇Vs ./ 3)
    Rτxy .= .-τxy .+      ηs_v .* (diff(Vxs, dims=2) ./ dy .+ diff(Vys, dims=1) ./ dx)
    τxx .+= Rτxx .* dτ_r
    τyy .+= Rτyy .* dτ_r
    τxy .+= Rτxy .* dτ_r
    # err
    err     = ifelse(mod(iter, nout) == 0, max(maximum(abs, RVx), maximum(abs, RVy), maximum(abs, ∇Vs)), err)
    mask    = (1:length(err_log)) .== ((iter - 1) ÷ nout + 1)
    err_log .= ifelse.(mask, err, err_log)
    return iter, err
end

function solve(Pt, Vxs, Vys, τxx, τyy, τxy, ∇Vs, RVx, RVy, Rτxx, Rτyy, Rτxy, ηs, ηs_v,
               dτ_r, νdτ, dτ_Pr, dx, dy, tol, maxiter, nout, iter, err, err_log)
    @trace while (iter < maxiter) & (err >= tol)
        iter, err = step(Pt, Vxs, Vys, τxx, τyy, τxy, ∇Vs, RVx, RVy, Rτxx, Rτyy, Rτxy, ηs, ηs_v,
                         dτ_r, νdτ, dτ_Pr, dx, dy, nout, iter, err, err_log)
    end
    return iter, err
end

function main(; nx=128, ny=128, backend=:auto, verbose=true)
    resolved, CRArray, CRNumber = init_backend(backend)
    use_reactant = resolved !== :none
    # independent physics
    ηs0     = 1.0     # reference shear viscosity
    P0      = 1.0     # reference pressure
    # dependent
    radius  = 0.1
    ηs_inc  = 1e-1    # inclusion shear viscosity
    ε̇       = 1.0     # background pure shear strain rate
    # dependent physics
    lx, ly  = 1.0, 1.0
    # numerics
    tol     = 1e-8
    maxiter = 40nx
    nout    = 2nx
    # preprocessing
    dx, dy  = lx / nx, ly / ny
    xc      = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc      = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    xv      = LinRange(-lx / 2         , lx / 2, nx+1)
    yv      = LinRange(-ly / 2         , ly / 2, ny+1)
    xce     = LinRange(-lx / 2 - dx / 2, lx / 2 + dx / 2, nx+2)
    yce     = LinRange(-ly / 2 - dy / 2, ly / 2 + dy / 2, ny+2)
    x2D_Vxs = xv    .+ 0*yce' # Ghost points
    y2D_Vys = 0*xce .+ yv'    # Ghost points
    # dmp solid
    re_m    = 5π
    r       = 0.5
    lτ_re   = min(lx, ly) / re_m
    vdτ     = min(dx, dy) / sqrt(4.1)
    θ_dτ    = lτ_re * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    νdτ     = vdτ * lτ_re
    dτ_Pr   = r / θ_dτ
    # initialisation
    Vxs     = CRArray(-ε̇ .* x2D_Vxs)
    Vys     = CRArray( ε̇ .* y2D_Vys)
    τxx     = CRArray(zeros(nx, ny))
    τyy     = CRArray(zeros(nx, ny))
    τxy     = CRArray(zeros(nx + 1, ny + 1))
    ∇Vs     = CRArray(zeros(nx, ny))
    RVx     = CRArray(zeros(nx - 1, ny))
    RVy     = CRArray(zeros(nx, ny - 1))
    Rτxx    = CRArray(zeros(nx, ny))
    Rτyy    = CRArray(zeros(nx, ny))
    Rτxy    = CRArray(zeros(nx + 1, ny + 1))
    # initial conditions
    Pt      = CRArray(fill(P0, nx, ny))
    ηs_v    = fill(ηs0, nx + 1, ny + 1)
    ηs_v[hypot.(xv, yv') .< radius] .= ηs_inc
    ηs_v    = CRArray(ηs_v)
    ηs      = CRArray(av(ηs_v))
    iter    = CRNumber(0)
    err     = CRNumber(10tol)
    err_log = CRArray(zeros(maxiter ÷ nout))
    # visualisation init
    fig = Figure(; size=(400, 600))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="P total"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Vxs"),
           Axis(fig[3, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Vys"))
    plt = (heatmap!(axs[1], xc,  yc, Array(Pt);  colorrange=(-3,3), colormap=(CairoMakie.Reverse(:matter), 1)),
           heatmap!(axs[2], xv,  yce, Array(Vxs); colormap=:turbo),
           heatmap!(axs[3], xce, yv,  Array(Vys); colormap=:turbo))
    cbs = (Colorbar(fig[1, 1][1, 2], plt[1]),
           Colorbar(fig[2, 1][1, 2], plt[2]),
           Colorbar(fig[3, 1][1, 2], plt[3]))
    hidexdecorations!.((axs[1], axs[2]))

    if use_reactant
        t_compile = time_ns()
        solve_react = @compile sync=true solve(Pt, Vxs, Vys, τxx, τyy, τxy, ∇Vs, RVx, RVy, Rτxx, Rτyy, Rτxy, ηs, ηs_v, dτ_r, νdτ, dτ_Pr, dx, dy, tol, maxiter, nout, iter, err, err_log)
        t_compile = (time_ns() - t_compile) / 1e9
        t_run = time_ns()
        iter, err = solve_react(Pt, Vxs, Vys, τxx, τyy, τxy, ∇Vs, RVx, RVy, Rτxx, Rτyy, Rτxy, ηs, ηs_v, dτ_r, νdτ, dτ_Pr, dx, dy, tol, maxiter, nout, iter, err, err_log)
        t_run = (time_ns() - t_run) / 1e9
        @printf "  compile: %.3f s, run: %.3f s\n" t_compile t_run
    else
        t_run = time_ns()
        iter, err = solve(Pt, Vxs, Vys, τxx, τyy, τxy, ∇Vs, RVx, RVy, Rτxx, Rτyy, Rτxy, ηs, ηs_v, dτ_r, νdτ, dτ_Pr, dx, dy, tol, maxiter, nout, iter, err, err_log)
        t_run = (time_ns() - t_run) / 1e9
        @printf "  run: %.3f s\n" t_run
    end
    @printf "  converged: iter/ny=%d, err=%1.3e\n" to_scalar(iter) ÷ ny to_scalar(err)
    if verbose
        for (i, e) in enumerate(Array(err_log))
            e == 0 && break  # stop at first unfilled slot
            @printf "  iter/ny=%d, err=%1.3e\n" i * nout ÷ ny e
        end
    end

    # visualisation
    plt[1][3] = Array(Pt) .- P0
    plt[2][3] = Array(Vxs .+ ε̇ .* x2D_Vxs)
    plt[3][3] = Array(Vys .- ε̇ .* y2D_Vys)
    display(fig)

    return
end

main(nx=128, ny=128, backend=:cpu, verbose=false)
