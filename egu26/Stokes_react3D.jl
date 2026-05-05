using Printf
using CairoMakie
using Reactant

include("reactant_helpers.jl")

@views avx(A) = @. 0.5 * (A[1:end-1, :, :] + A[2:end, :, :])
@views avy(A) = @. 0.5 * (A[:, 1:end-1, :] + A[:, 2:end, :])
@views avz(A) = @. 0.5 * (A[:, :, 1:end-1] + A[:, :, 2:end])
# vertex (8-point) average to cell centres
@views av(A)  = @. 0.125 * (A[1:end-1,1:end-1,1:end-1] + A[1:end-1,1:end-1,2:end] +
                              A[1:end-1,2:end,  1:end-1] + A[1:end-1,2:end,  2:end] +
                              A[2:end,  1:end-1,1:end-1] + A[2:end,  1:end-1,2:end] +
                              A[2:end,  2:end,  1:end-1] + A[2:end,  2:end,  2:end])
# edge averages for off-diagonal stress vertices
@views avxyi(A) = @. 0.25 * (A[1:end-1,1:end-1,2:end-1] + A[1:end-1,2:end,2:end-1] + A[2:end,1:end-1,2:end-1] + A[2:end,2:end,2:end-1])
@views avxzi(A) = @. 0.25 * (A[1:end-1,2:end-1,1:end-1] + A[1:end-1,2:end-1,2:end] + A[2:end,2:end-1,1:end-1] + A[2:end,2:end-1,2:end])
@views avyzi(A) = @. 0.25 * (A[2:end-1,1:end-1,1:end-1] + A[2:end-1,1:end-1,2:end] + A[2:end-1,2:end,1:end-1] + A[2:end-1,2:end,2:end])

@views function step3D(Pt, Vxs, Vys, Vzs,
                       τxx, τyy, τzz, τxy, τxz, τyz,
                       ∇Vs, RVx, RVy, RVz,
                       Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                       ηs, ηs_xy, ηs_xz, ηs_yz,
                       dτ_r, νdτ, dτ_Pr, dx, dy, dz, nout, iter, err, err_log)
    iter += 1
    # divergence
    ∇Vs .= diff(Vxs[:, 2:end-1, 2:end-1], dims=1) ./ dx .+
           diff(Vys[2:end-1, :, 2:end-1], dims=2) ./ dy .+
           diff(Vzs[2:end-1, 2:end-1, :], dims=3) ./ dz
    Pt .-= ∇Vs .* ηs .* dτ_Pr
    # velocity residuals
    RVx .= diff(.-Pt .+ τxx, dims=1) ./ dx .+
           diff(τxy[2:end-1, :, :], dims=2)   ./ dy .+    # τxy: (nx+1, ny+1, nz)
           diff(τxz[2:end-1, :, :], dims=3)   ./ dz       # τxz: (nx+1, ny, nz+1)
    RVy .= diff(.-Pt .+ τyy, dims=2) ./ dy .+
           diff(τxy[:, 2:end-1, :], dims=1)   ./ dx .+
           diff(τyz[:, 2:end-1, :], dims=3)   ./ dz       # τyz: (nx, ny+1, nz+1)
    RVz .= diff(.-Pt .+ τzz, dims=3) ./ dz .+
           diff(τxz[:, :, 2:end-1], dims=1)   ./ dx .+
           diff(τyz[:, :, 2:end-1], dims=2)   ./ dy
    Vxs[2:end-1, 2:end-1, 2:end-1] .+= RVx .* νdτ ./ avx(ηs)
    Vys[2:end-1, 2:end-1, 2:end-1] .+= RVy .* νdτ ./ avy(ηs)
    Vzs[2:end-1, 2:end-1, 2:end-1] .+= RVz .* νdτ ./ avz(ηs)
    # BC free slip on all faces
    Vxs[:, [1, end], :] .= Vxs[:, [2, end-1], :]
    Vxs[:, :, [1, end]] .= Vxs[:, :, [2, end-1]]
    Vys[[1, end], :, :] .= Vys[[2, end-1], :, :]
    Vys[:, :, [1, end]] .= Vys[:, :, [2, end-1]]
    Vzs[[1, end], :, :] .= Vzs[[2, end-1], :, :]
    Vzs[:, [1, end], :] .= Vzs[:, [2, end-1], :]
    # stress residuals
    Rτxx .= .-τxx .+ 2.0 .* ηs .* (diff(Vxs[:, 2:end-1, 2:end-1], dims=1) ./ dx .- ∇Vs ./ 3)
    Rτyy .= .-τyy .+ 2.0 .* ηs .* (diff(Vys[2:end-1, :, 2:end-1], dims=2) ./ dy .- ∇Vs ./ 3)
    Rτzz .= .-τzz .+ 2.0 .* ηs .* (diff(Vzs[2:end-1, 2:end-1, :], dims=3) ./ dz .- ∇Vs ./ 3)
    Rτxy .= .-τxy .+ ηs_xy .* (diff(Vxs[:, :, 2:end-1], dims=2) ./ dy .+ diff(Vys[:, :, 2:end-1], dims=1) ./ dx)
    Rτxz .= .-τxz .+ ηs_xz .* (diff(Vxs[:, 2:end-1, :], dims=3) ./ dz .+ diff(Vzs[:, 2:end-1, :], dims=1) ./ dx)
    Rτyz .= .-τyz .+ ηs_yz .* (diff(Vys[2:end-1, :, :], dims=3) ./ dz .+ diff(Vzs[2:end-1, :, :], dims=2) ./ dy)
    τxx .+= Rτxx .* dτ_r
    τyy .+= Rτyy .* dτ_r
    τzz .+= Rτzz .* dτ_r
    τxy .+= Rτxy .* dτ_r
    τxz .+= Rτxz .* dτ_r
    τyz .+= Rτyz .* dτ_r
    # error
    err     = ifelse(mod(iter, nout) == 0,
                     max(maximum(abs, RVx), maximum(abs, RVy), maximum(abs, RVz), maximum(abs, ∇Vs)), err)
    mask    = (1:length(err_log)) .== ((iter - 1) ÷ nout + 1)
    err_log .= ifelse.(mask, err, err_log)
    return iter, err
end

function solve3D(Pt, Vxs, Vys, Vzs,
                 τxx, τyy, τzz, τxy, τxz, τyz,
                 ∇Vs, RVx, RVy, RVz,
                 Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                 ηs, ηs_xy, ηs_xz, ηs_yz,
                 dτ_r, νdτ, dτ_Pr, dx, dy, dz, tol, maxiter, nout, iter, err, err_log)
    @trace while (iter < maxiter) & (err >= tol)
        iter, err = step3D(Pt, Vxs, Vys, Vzs,
                           τxx, τyy, τzz, τxy, τxz, τyz,
                           ∇Vs, RVx, RVy, RVz,
                           Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                           ηs, ηs_xy, ηs_xz, ηs_yz,
                           dτ_r, νdτ, dτ_Pr, dx, dy, dz, nout, iter, err, err_log)
    end
    return iter, err
end

function main(; nx=64, ny=64, nz=64, backend=:auto, verbose=true, do_plot=true, bench=false)
    resolved, CRArray, CRNumber = init_backend(backend)
    use_reactant = resolved !== :none
    # physics — pure shear in xz plane, y is the neutral axis
    ηs0    = 1.0
    P0     = 1.0
    radius = 0.1
    ηs_inc = 1e-1
    ε̇      = 1.0    # background pure-shear strain rate (ε̇xx = -ε̇, ε̇zz = +ε̇, ε̇yy = 0)
    lx, ly, lz = 1.0, 1.0, 1.0
    # numerics
    tol     = 1e-8
    maxiter = 40nx
    nout    = 2nx
    # preprocessing
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    xc  = LinRange(-lx/2 + dx/2, lx/2 - dx/2, nx)
    yc  = LinRange(-ly/2 + dy/2, ly/2 - dy/2, ny)
    zc  = LinRange(-lz/2 + dz/2, lz/2 - dz/2, nz)
    xv  = LinRange(-lx/2, lx/2, nx+1)
    yv  = LinRange(-ly/2, ly/2, ny+1)
    zv  = LinRange(-lz/2, lz/2, nz+1)
    xce = LinRange(-lx/2 - dx/2, lx/2 + dx/2, nx+2)
    yce = LinRange(-ly/2 - dy/2, ly/2 + dy/2, ny+2)
    zce = LinRange(-lz/2 - dz/2, lz/2 + dz/2, nz+2)
    # dmp
    re_m  = 6π
    r     = 0.95
    lτ_re = min(lx, ly, lz) / re_m
    vdτ   = min(dx, dy, dz) / sqrt(6.1)   # 3D stability: /sqrt(ndim+2)
    θ_dτ  = lτ_re * (r + 4/3) / vdτ
    dτ_r  = 1.0 / (θ_dτ + 1.0)
    νdτ   = vdτ * lτ_re
    dτ_Pr = r / θ_dτ
    # initial velocity — pure shear: compress in x, extend in z, neutral in y
    Vxs_init = [-ε̇ * x for x in xv, _ in yce, _ in zce]
    Vys_init = zeros(length(xce), length(yv), length(zce))
    Vzs_init = [ ε̇ * z for _ in xce, _ in yce, z in zv]
    # wrap
    Vxs  = CRArray(Vxs_init)
    Vys  = CRArray(Vys_init)
    Vzs  = CRArray(Vzs_init)
    τxx  = CRArray(zeros(nx, ny, nz))
    τyy  = CRArray(zeros(nx, ny, nz))
    τzz  = CRArray(zeros(nx, ny, nz))
    τxy  = CRArray(zeros(nx+1, ny+1, nz))   # vertex in x,y; cell in z
    τxz  = CRArray(zeros(nx+1, ny, nz+1))   # vertex in x,z; cell in y
    τyz  = CRArray(zeros(nx, ny+1, nz+1))   # vertex in y,z; cell in x
    ∇Vs  = CRArray(zeros(nx, ny, nz))
    RVx  = CRArray(zeros(nx-1, ny, nz))
    RVy  = CRArray(zeros(nx, ny-1, nz))
    RVz  = CRArray(zeros(nx, ny, nz-1))
    Rτxx = CRArray(zeros(nx, ny, nz))
    Rτyy = CRArray(zeros(nx, ny, nz))
    Rτzz = CRArray(zeros(nx, ny, nz))
    Rτxy = CRArray(zeros(nx+1, ny+1, nz))
    Rτxz = CRArray(zeros(nx+1, ny, nz+1))
    Rτyz = CRArray(zeros(nx, ny+1, nz+1))
    Pt   = CRArray(fill(P0, nx, ny, nz))
    # viscosity: sphere inclusion centred at origin
    ηs_v_h = fill(ηs0, nx+2, ny+2, nz+2)
    for k in eachindex(xce), j in eachindex(yce), i in eachindex(xce)
        hypot(xce[i], yce[j], xce[k]) < radius && (ηs_v_h[i,j,k] = ηs_inc)
    end
    ηs    = CRArray(ηs_v_h[2:end-1, 2:end-1, 2:end-1])
    ηs_xy = CRArray(avxyi(ηs_v_h))   # average to (nx+1, ny+1, nz) τxy nodes
    ηs_xz = CRArray(avxzi(ηs_v_h))   # average to (nx+1, ny, nz+1) τxz nodes
    ηs_yz = CRArray(avyzi(ηs_v_h))   # average to (nx, ny+1, nz+1) τyz nodes
    iter    = CRNumber(0)
    err     = CRNumber(10tol)
    err_log = CRArray(zeros(maxiter ÷ nout))
    _arrs   = (Pt, Vxs, Vys, Vzs, τxx, τyy, τzz, τxy, τxz, τyz,
               ∇Vs, RVx, RVy, RVz, Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
               ηs, ηs_xy, ηs_xz, ηs_yz)
    A_bytes = sum(A -> length(A) * sizeof(eltype(A)), _arrs)
    n_arr   = length(_arrs)
    t_compile = 0.0; t_run = 0.0

    # visualisation init — xz mid-plane slices (j = ny÷2)
    if do_plot
        out_dir = "output"; mkpath(out_dir)
        jmid = ny ÷ 2
        fig = Figure(; size=(400, 600))
        axs = (Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="P total  (xz mid)"),
               Axis(fig[2,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vxs (xz mid)"),
               Axis(fig[3,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vzs (xz mid)"))
        plt = (heatmap!(axs[1], xc, zc,  Array(Pt)[:, jmid, :];         colorrange=(-3,3), colormap=(CairoMakie.Reverse(:matter), 1)),
               heatmap!(axs[2], xv, zce, Array(Vxs_init)[:, jmid+1, :]; colormap=:turbo),
               heatmap!(axs[3], xce, zv, Array(Vzs_init)[jmid+1, :, :]; colormap=:turbo))
        cbs = (Colorbar(fig[1,1][1,2], plt[1]),
               Colorbar(fig[2,1][1,2], plt[2]),
               Colorbar(fig[3,1][1,2], plt[3]))
        hidexdecorations!.((axs[1], axs[2]))
    end
    if use_reactant
        t_compile = time_ns()
        solve_react = @compile sync=true solve3D(Pt, Vxs, Vys, Vzs,
                                                  τxx, τyy, τzz, τxy, τxz, τyz,
                                                  ∇Vs, RVx, RVy, RVz,
                                                  Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                                                  ηs, ηs_xy, ηs_xz, ηs_yz,
                                                  dτ_r, νdτ, dτ_Pr, dx, dy, dz,
                                                  tol, maxiter, nout, iter, err, err_log)
        t_compile = (time_ns() - t_compile) / 1e9
        t_run = time_ns()
        iter, err = solve_react(Pt, Vxs, Vys, Vzs,
                                τxx, τyy, τzz, τxy, τxz, τyz,
                                ∇Vs, RVx, RVy, RVz,
                                Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                                ηs, ηs_xy, ηs_xz, ηs_yz,
                                dτ_r, νdτ, dτ_Pr, dx, dy, dz,
                                tol, maxiter, nout, iter, err, err_log)
        t_run = (time_ns() - t_run) / 1e9
        @printf "  compile: %.3f s, run: %.3f s\n" t_compile t_run
    else
        t_run = time_ns()
        iter, err = solve3D(Pt, Vxs, Vys, Vzs,
                            τxx, τyy, τzz, τxy, τxz, τyz,
                            ∇Vs, RVx, RVy, RVz,
                            Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                            ηs, ηs_xy, ηs_xz, ηs_yz,
                            dτ_r, νdτ, dτ_Pr, dx, dy, dz,
                            tol, maxiter, nout, iter, err, err_log)
        t_run = (time_ns() - t_run) / 1e9
        @printf "  run: %.3f s\n" t_run
    end
    @printf "  converged: iter/nz=%d, err=%1.3e\n" to_scalar(iter) ÷ nz to_scalar(err)
    if bench
        niter = to_scalar(iter)
        T_eff = 2 * A_bytes * 1e-9 * niter / t_run
        @printf "  T_eff=%.2f GB/s  (nx=%d, %d arrays, niter=%d)\n" T_eff nx n_arr niter
        return (; t_compile, t_run, niter, T_eff)
    end
    if verbose
        for (i, e) in enumerate(Array(err_log))
            e == 0 && break
            @printf "  iter/nz=%d, err=%1.3e\n" i * nout ÷ nz e
        end
    end

    # visualisation — xz mid-plane
    if do_plot
        plt[1][3] = (Array(Pt) .- P0)[:, jmid, :]
        plt[2][3] = Array(Vxs)[:, jmid+1, :]
        plt[3][3] = Array(Vzs)[jmid+1, :, :]
        # display(fig)
        save(joinpath(out_dir, "output_Stokes3D.png"), fig)
    end

    return
end

res = 128
isdefined(Main, :_bench_sweep) || main(nx=res, ny=res, nz=res, backend=:auto, verbose=false, do_plot=true)
