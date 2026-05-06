using LinearAlgebra, Statistics, Printf
#using CairoMakie
using Reactant

include("reactant_helpers.jl")

# inner region (remove one ghost on each side)
@views inn(A)  = A[2:end-1, 2:end-1, 2:end-1]
@views innx(A) = A[2:end-1, :, :]
@views inny(A) = A[:, 2:end-1, :]
@views innz(A) = A[:, :, 2:end-1]
# face-to-centre averages
@views avx(A)  = @. 0.5 * (A[1:end-1, :, :] + A[2:end, :, :])
@views avy(A)  = @. 0.5 * (A[:, 1:end-1, :] + A[:, 2:end, :])
@views avz(A)  = @. 0.5 * (A[:, :, 1:end-1] + A[:, :, 2:end])
# staggered interior averages for solid flux (remove ghosts from non-staggered dims)
@views avxi(A) = @. 0.5 * (A[1:end-1, 2:end-1, 2:end-1] + A[2:end, 2:end-1, 2:end-1])
@views avyi(A) = @. 0.5 * (A[2:end-1, 1:end-1, 2:end-1] + A[2:end-1, 2:end, 2:end-1])
@views avzi(A) = @. 0.5 * (A[2:end-1, 2:end-1, 1:end-1] + A[2:end-1, 2:end-1, 2:end])
# local max over 6-neighbours for adaptive Darcy dt
@views function maxloc(A)
    max.(A[2:end-1, 2:end-1, 2:end-1],
         max.(max.(A[1:end-2, 2:end-1, 2:end-1], A[3:end, 2:end-1, 2:end-1]),
         max.(max.(A[2:end-1, 1:end-2, 2:end-1], A[2:end-1, 3:end, 2:end-1]),
              max.(A[2:end-1, 2:end-1, 1:end-2], A[2:end-1, 2:end-1, 3:end]))))
end
# Neumann BCs on all 6 faces
@views function neumann_bcs_ap!(A)
    A[1, :, :]   .= A[2, :, :]
    A[end, :, :] .= A[end-1, :, :]
    A[:, 1, :]   .= A[:, 2, :]
    A[:, end, :] .= A[:, end-1, :]
    A[:, :, 1]   .= A[:, :, 2]
    A[:, :, end] .= A[:, :, end-1]
end

@views function step3Dpw(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                          Vxs, Vys, Vzs,
                          τxx, τyy, τzz, τxy, τxz, τyz,
                          ∇Vs, qDx, qDy, qDz, Qfx, Qfy, Qfz, dϕdt,
                          lc_loc, re, θ_dτ, dτ_βf,
                          RVx, RVy, RVz, RPt, RPf,
                          Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                          RqDx, RqDy, RqDz,
                          ϕ_old, ρt_old,
                          dx, dy, dz, dt, nout,
                          ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                          νdτ, dτ_r, dτ_Pr, cfl, lx, ly, lz,
                          iter, err, err_log)
    iter += 1
    # material params
    ηϕ   .= ηϕ_bg .* (ϕ_bg ./ ϕ) .* (1.0 .+ 0.5 .* (1.0 / R - 1.0) .* (1.0 .+ tanh.(Pe ./ λPe)))
    k_ηf .= exp.((1.0 - ϕ_rel) .* log.(k_ηf) .+ ϕ_rel .* log.(k_ηf0 .* (ϕ ./ ϕ_bg) .^ npow))
    ρt   .= (1.0 .- ϕ) .* ρs .+ ϕ .* ρf
    # divergence of solid velocity (interior)
    ∇Vs  .= diff(Vxs[:, 2:end-1, 2:end-1], dims=1) ./ dx .+
            diff(Vys[2:end-1, :, 2:end-1], dims=2) ./ dy .+
            diff(Vzs[2:end-1, 2:end-1, :], dims=3) ./ dz
    # solid velocity residuals (gravity in -z: ρt appears in RVz)
    RVx  .= diff(.-Pt[:, 2:end-1, 2:end-1] .+ τxx[:, 2:end-1, 2:end-1], dims=1) ./ dx .+
            diff(τxy, dims=2) ./ dy .+ diff(τxz, dims=3) ./ dz
    RVy  .= diff(.-Pt[2:end-1, :, 2:end-1] .+ τyy[2:end-1, :, 2:end-1], dims=2) ./ dy .+
            diff(τxy, dims=1) ./ dx .+ diff(τyz, dims=3) ./ dz
    RVz  .= diff(.-Pt[2:end-1, 2:end-1, :] .+ τzz[2:end-1, 2:end-1, :], dims=3) ./ dz .+
            diff(τxz, dims=1) ./ dx .+ diff(τyz, dims=2) ./ dy .- avz(ρt[2:end-1, 2:end-1, :])
    Vxs[:, 2:end-1, 2:end-1] .+= RVx .* νdτ ./ ηs
    Vys[2:end-1, :, 2:end-1] .+= RVy .* νdτ ./ ηs
    Vzs[2:end-1, 2:end-1, :] .+= RVz .* νdτ ./ ηs
    # stress residuals
    Rτxx[2:end-1, 2:end-1, 2:end-1] .= .-inn(τxx) .+ 2.0 .* ηs .* (diff(Vxs[:, 2:end-1, 2:end-1], dims=1) ./ dx .- ∇Vs ./ 3)
    Rτyy[2:end-1, 2:end-1, 2:end-1] .= .-inn(τyy) .+ 2.0 .* ηs .* (diff(Vys[2:end-1, :, 2:end-1], dims=2) ./ dy .- ∇Vs ./ 3)
    Rτzz[2:end-1, 2:end-1, 2:end-1] .= .-inn(τzz) .+ 2.0 .* ηs .* (diff(Vzs[2:end-1, 2:end-1, :], dims=3) ./ dz .- ∇Vs ./ 3)
    Rτxy .= .-τxy .+ ηs .* (diff(Vxs[:, :, 2:end-1], dims=2) ./ dy .+ diff(Vys[:, :, 2:end-1], dims=1) ./ dx)
    Rτxz .= .-τxz .+ ηs .* (diff(Vxs[:, 2:end-1, :], dims=3) ./ dz .+ diff(Vzs[:, 2:end-1, :], dims=1) ./ dx)
    Rτyz .= .-τyz .+ ηs .* (diff(Vys[2:end-1, :, :], dims=3) ./ dz .+ diff(Vzs[2:end-1, :, :], dims=2) ./ dy)
    τxx .+= Rτxx .* dτ_r
    τyy .+= Rτyy .* dτ_r
    τzz .+= Rτzz .* dτ_r
    τxy .+= Rτxy .* dτ_r
    τxz .+= Rτxz .* dτ_r
    τyz .+= Rτyz .* dτ_r
    # total pressure from solid mass balance
    RPt .= .-inn(ϕ .- ϕ_old) ./ dt .+
           diff((1.0 .- avxi(ϕ)) .* Vxs[:, 2:end-1, 2:end-1], dims=1) ./ dx .+
           diff((1.0 .- avyi(ϕ)) .* Vys[2:end-1, :, 2:end-1], dims=2) ./ dy .+
           diff((1.0 .- avzi(ϕ)) .* Vzs[2:end-1, 2:end-1, :], dims=3) ./ dz
    # porous flow (adaptive τ based on local compaction length)
    lc_loc .= sqrt.(k_ηf .* ηϕ)
    re     .= π .+ sqrt.(π^2 .+ (min(lx, ly, lz) ./ lc_loc) .^ 2)
    θ_dτ   .= min(lx, ly, lz) ./ re ./ cfl ./ max(dx, dy, dz)
    dτ_βf  .= cfl * min(lx, ly, lz) .* max(dx, dy, dz) ./ maxloc(re .* k_ηf)
    RqDx   .= .-qDx .- avxi(k_ηf) .*  diff(Pf[:, 2:end-1, 2:end-1], dims=1) ./ dx
    RqDy   .= .-qDy .- avyi(k_ηf) .*  diff(Pf[2:end-1, :, 2:end-1], dims=2) ./ dy
    RqDz   .= .-qDz .- avzi(k_ηf) .* (diff(Pf[2:end-1, 2:end-1, :], dims=3) ./ dz .- ρf)
    qDx   .+= RqDx ./ (1.0 .+ avxi(θ_dτ))
    qDy   .+= RqDy ./ (1.0 .+ avyi(θ_dτ))
    qDz   .+= RqDz ./ (1.0 .+ avzi(θ_dτ))
    Qfx    .= ρf .* qDx .+ avxi(ρt) .* Vxs[:, 2:end-1, 2:end-1]
    Qfy    .= ρf .* qDy .+ avyi(ρt) .* Vys[2:end-1, :, 2:end-1]
    Qfz    .= ρf .* qDz .+ avzi(ρt) .* Vzs[2:end-1, 2:end-1, :]
    Pt[2:end-1, 2:end-1, 2:end-1] .-= RPt .* ηs .* (1.0 .- inn(ϕ)) .* dτ_Pr
    # fluid pressure from total mass balance
    RPf .= inn(ρt .- ρt_old) ./ dt .+
           diff(Qfx, dims=1) ./ dx .+
           diff(Qfy, dims=2) ./ dy .+
           diff(Qfz, dims=3) ./ dz
    Pf[2:end-1, 2:end-1, 2:end-1] .-= RPf .* inn(ϕ) .* dτ_βf
    # porosity update
    Pe   .= Pf .- Pt
    dϕdt .= inn(Pe ./ ηϕ)
    ϕ[2:end-1, 2:end-1, 2:end-1] .= inn(ϕ_old) .+ dt .* dϕdt
    neumann_bcs_ap!(ϕ)
    # error
    err  = ifelse(mod(iter, nout) == 0,
                  max(maximum(abs, RVx), maximum(abs, RVy), maximum(abs, RVz),
                      maximum(abs, RPt), maximum(abs, RPf)),
                  err)
    mask    = (1:length(err_log)) .== ((iter - 1) ÷ nout + 1)
    err_log .= ifelse.(mask, err, err_log)
    return iter, err
end

function solve3Dpw(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                   Vxs, Vys, Vzs,
                   τxx, τyy, τzz, τxy, τxz, τyz,
                   ∇Vs, qDx, qDy, qDz, Qfx, Qfy, Qfz, dϕdt,
                   lc_loc, re, θ_dτ, dτ_βf,
                   RVx, RVy, RVz, RPt, RPf,
                   Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                   RqDx, RqDy, RqDz,
                   ϕ_old, ρt_old,
                   dx, dy, dz, dt, nout,
                   ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                   νdτ, dτ_r, dτ_Pr, cfl, lx, ly, lz,
                   tol, maxiter, iter, err, err_log)
    ϕ_old  .= ϕ
    ρt_old .= (1.0 .- ϕ) .* ρs .+ ϕ .* ρf
    @trace while (iter < maxiter) & (err >= tol)
        iter, err = step3Dpw(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                              Vxs, Vys, Vzs,
                              τxx, τyy, τzz, τxy, τxz, τyz,
                              ∇Vs, qDx, qDy, qDz, Qfx, Qfy, Qfz, dϕdt,
                              lc_loc, re, θ_dτ, dτ_βf,
                              RVx, RVy, RVz, RPt, RPf,
                              Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                              RqDx, RqDy, RqDz,
                              ϕ_old, ρt_old,
                              dx, dy, dz, dt, nout,
                              ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                              νdτ, dτ_r, dτ_Pr, cfl, lx, ly, lz,
                              iter, err, err_log)
    end
    return iter, err
end

function main(; nx=32, nt=1, backend=:auto, verbose=true, do_plot=true, bench=false)
    resolved, CRArray, CRNumber = init_backend(backend)
    use_reactant = resolved !== :none
    # physics
    lc    = 1.0
    Δρg   = 1.0
    ηϕ_bg = 1.0
    psc   = Δρg * lc
    tsc   = ηϕ_bg / psc
    R     = 100.0
    lz_lc = 80.0 / sqrt(0.9R)   # z is the vertical (long) dimension
    w_lc  = 5.0 / sqrt(0.9R)
    ρs_ρf = 1.2
    ϕ_bg  = 0.005
    ϕ_A   = 3.0
    npow  = 3.0
    ra    = 2.0                  # lz / lx aspect ratio
    λPe   = 0.01 * psc
    ηs    = ηϕ_bg * ϕ_bg
    lz    = lz_lc * lc
    lx    = lz / ra
    ly    = lx                   # square horizontal cross-section
    w     = w_lc * lc
    k_ηf0 = lc^2 / ηϕ_bg
    dt0   = 1e-6 * tsc
    # numerics
    ny      = nx
    nz      = ceil(Int, nx * ra)
    tol     = 1e-3
    cfl_dt  = 2e-2
    ϕ_rel   = 1e-1
    nt      = 1
    maxiter = 150nz
    nout    = 5nz
    nviz    = 5
    top     = 0.9nz
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    xc      = LinRange(-lx/2 + dx/2, lx/2 - dx/2, nx)
    yc      = LinRange(-ly/2 + dy/2, ly/2 - dy/2, ny)
    zc      = LinRange(-lz/2 + dz/2, lz/2 - dz/2, nz)
    ρf      = Δρg / (ρs_ρf - 1.0)
    ρs      = ρf * ρs_ρf
    # dmp solid
    re_m    = 3.5π
    r       = 0.8
    lτ_re_m = min(lx, ly, lz) / re_m
    vdτ     = min(dx, dy, dz) / sqrt(6.1)
    θ_dτ_s  = lτ_re_m * (r + 4/3) / vdτ
    dτ_r    = 1.0 / (θ_dτ_s + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ_s
    cfl     = 1 / sqrt(3.1)
    # initial conditions — porosity plume in x-y plane, rising in z
    ϕ_h    = ϕ_bg .* (1 .+ ϕ_A .* exp.(-(xc ./ w).^2 .- (yc' ./ w).^2 .+ zeros(1,1,nz) .- ((reshape(zc, 1,1,nz) .+ lz/4) ./ w).^2))
    ρt_ref = (1.0 - ϕ_bg) * ρs + ϕ_bg * ρf
    Pt_h   = -ρt_ref .* (zeros(nx,ny) .+ reshape(zc, 1,1,nz))   # hydrostatic in z
    # wrap
    ϕ      = CRArray(ϕ_h)
    k_ηf   = CRArray(k_ηf0 .* (ϕ_h ./ ϕ_bg) .^ npow)
    ρt     = CRArray((1.0 .- ϕ_h) .* ρs .+ ϕ_h .* ρf)
    Pt     = CRArray(Pt_h)
    Pf     = CRArray(Pt_h)
    Pe     = CRArray(zeros(nx, ny, nz))
    ϕ_old  = CRArray(ϕ_h)
    ρt_old = CRArray((1.0 .- ϕ_h) .* ρs .+ ϕ_h .* ρf)
    ηϕ     = CRArray(zeros(nx, ny, nz))
    Vxs    = CRArray(zeros(nx-1, ny,   nz  ))
    Vys    = CRArray(zeros(nx,   ny-1, nz  ))
    Vzs    = CRArray(zeros(nx,   ny,   nz-1))
    τxx    = CRArray(zeros(nx, ny, nz))
    τyy    = CRArray(zeros(nx, ny, nz))
    τzz    = CRArray(zeros(nx, ny, nz))
    τxy    = CRArray(zeros(nx-1, ny-1, nz-2))
    τxz    = CRArray(zeros(nx-1, ny-2, nz-1))
    τyz    = CRArray(zeros(nx-2, ny-1, nz-1))
    ∇Vs    = CRArray(zeros(nx-2, ny-2, nz-2))
    dϕdt   = CRArray(zeros(nx-2, ny-2, nz-2))
    qDx    = CRArray(zeros(nx-1, ny-2, nz-2))
    qDy    = CRArray(zeros(nx-2, ny-1, nz-2))
    qDz    = CRArray(zeros(nx-2, ny-2, nz-1))
    Qfx    = CRArray(zeros(nx-1, ny-2, nz-2))
    Qfy    = CRArray(zeros(nx-2, ny-1, nz-2))
    Qfz    = CRArray(zeros(nx-2, ny-2, nz-1))
    RVx    = CRArray(zeros(nx-1, ny-2, nz-2))
    RVy    = CRArray(zeros(nx-2, ny-1, nz-2))
    RVz    = CRArray(zeros(nx-2, ny-2, nz-1))
    RPt    = CRArray(zeros(nx-2, ny-2, nz-2))
    RPf    = CRArray(zeros(nx-2, ny-2, nz-2))
    Rτxx   = CRArray(zeros(nx, ny, nz))
    Rτyy   = CRArray(zeros(nx, ny, nz))
    Rτzz   = CRArray(zeros(nx, ny, nz))
    Rτxy   = CRArray(zeros(nx-1, ny-1, nz-2))
    Rτxz   = CRArray(zeros(nx-1, ny-2, nz-1))
    Rτyz   = CRArray(zeros(nx-2, ny-1, nz-1))
    RqDx   = CRArray(zeros(nx-1, ny-2, nz-2))
    RqDy   = CRArray(zeros(nx-2, ny-1, nz-2))
    RqDz   = CRArray(zeros(nx-2, ny-2, nz-1))
    lc_loc = CRArray(zeros(nx, ny, nz))
    re     = CRArray(zeros(nx, ny, nz))
    θ_dτ   = CRArray(zeros(nx, ny, nz))
    dτ_βf  = CRArray(zeros(nx-2, ny-2, nz-2))
    dt_r   = CRNumber(dt0)
    iter   = CRNumber(0)
    err    = CRNumber(10tol)
    err_log = CRArray(zeros(maxiter ÷ nout))
    _arrs   = (ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf, ϕ_old, ρt_old,
               Vxs, Vys, Vzs, τxx, τyy, τzz, τxy, τxz, τyz, ∇Vs, dϕdt,
               qDx, qDy, qDz, Qfx, Qfy, Qfz, RVx, RVy, RVz, RPt, RPf,
               Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz, RqDx, RqDy, RqDz,
               lc_loc, re, θ_dτ, dτ_βf)
    A_bytes = sum(A -> length(A) * sizeof(eltype(A)), _arrs)
    n_arr   = length(_arrs)
    t_compile = 0.0; t_run = 0.0

    # visualisation init — xz mid-plane slice (j = ny÷2)
    if do_plot
        out_dir = "output"; mkpath(out_dir)
        jmid = ny ÷ 2
        fig = Figure(; size=(600, 800))
        axs = (Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Porosity (xz mid)"),
               Axis(fig[1,2][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Pe (xz mid)"),
               Axis(fig[2,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="permeability (xz mid)"),
               Axis(fig[2,2][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="bulk viscosity (xz mid)"))
        plt = (heatmap!(axs[1], xc, zc, Array(ϕ)[:, jmid, :];        colormap=:turbo),
               heatmap!(axs[2], xc, zc, Array(Pf .- Pt)[:, jmid, :]; colormap=:turbo),
               heatmap!(axs[3], xc, zc, Array(k_ηf)[:, jmid, :];     colormap=:turbo),
               heatmap!(axs[4], xc, zc, Array(ηϕ)[:, jmid, :];       colormap=:turbo))
        cbs = (Colorbar(fig[1,1][1,2], plt[1]),
               Colorbar(fig[1,2][1,2], plt[2]),
               Colorbar(fig[2,1][1,2], plt[3]),
               Colorbar(fig[2,2][1,2], plt[4]))
        hideydecorations!.((axs[2], axs[4]))
    end
    nxm = (nx + 1) ÷ 2
    nym = (ny + 1) ÷ 2
    time_evo = [0.0]
    ϕmax_evo = [to_scalar(maximum(ϕ))]

    # compile once before time loop
    if use_reactant
        t_compile = time_ns()
        solve_react = @compile sync=true solve3Dpw(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                                                    Vxs, Vys, Vzs,
                                                    τxx, τyy, τzz, τxy, τxz, τyz,
                                                    ∇Vs, qDx, qDy, qDz, Qfx, Qfy, Qfz, dϕdt,
                                                    lc_loc, re, θ_dτ, dτ_βf,
                                                    RVx, RVy, RVz, RPt, RPf,
                                                    Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                                                    RqDx, RqDy, RqDz,
                                                    ϕ_old, ρt_old,
                                                    dx, dy, dz, dt_r, nout,
                                                    ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                                                    νdτ, dτ_r, dτ_Pr, cfl, lx, ly, lz,
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
                                    Vxs, Vys, Vzs,
                                    τxx, τyy, τzz, τxy, τxz, τyz,
                                    ∇Vs, qDx, qDy, qDz, Qfx, Qfy, Qfz, dϕdt,
                                    lc_loc, re, θ_dτ, dτ_βf,
                                    RVx, RVy, RVz, RPt, RPf,
                                    Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                                    RqDx, RqDy, RqDz,
                                    ϕ_old, ρt_old,
                                    dx, dy, dz, dt_r, nout,
                                    ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                                    νdτ, dτ_r, dτ_Pr, cfl, lx, ly, lz,
                                    tol, maxiter, iter, err, err_log)
            t_run = (time_ns() - t_run) / 1e9
            @printf "  run: %.3f s\n" t_run
        else
            t_run = time_ns()
            iter, err = solve3Dpw(ϕ, k_ηf, ηϕ, ρt, Pe, Pt, Pf,
                                  Vxs, Vys, Vzs,
                                  τxx, τyy, τzz, τxy, τxz, τyz,
                                  ∇Vs, qDx, qDy, qDz, Qfx, Qfy, Qfz, dϕdt,
                                  lc_loc, re, θ_dτ, dτ_βf,
                                  RVx, RVy, RVz, RPt, RPf,
                                  Rτxx, Rτyy, Rτzz, Rτxy, Rτxz, Rτyz,
                                  RqDx, RqDy, RqDz,
                                  ϕ_old, ρt_old,
                                  dx, dy, dz, dt_r, nout,
                                  ρf, ρs, ηs, ηϕ_bg, ϕ_bg, npow, λPe, R, k_ηf0, ϕ_rel,
                                  νdτ, dτ_r, dτ_Pr, cfl, lx, ly, lz,
                                  tol, maxiter, iter, err, err_log)
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
                @printf "  iter/nz=%d, errs=%1.3e\n" i * nout ÷ nz e
            end
        end

        # diagnostic monitoring
        push!(time_evo, time_evo[end] + to_scalar(dt_r))
        push!(ϕmax_evo, to_scalar(maximum(ϕ)))
        dt_r = CRNumber(cfl_dt * to_scalar(maximum(ϕ)) / maximum(abs, Array(dϕdt)))

        # visualisation — xz mid-plane slice
        if do_plot
            if mod(it, nviz) == 0 || it == 1
                plt[1][3] = Array(ϕ)[:, jmid, :]
                plt[2][3] = Array(Pe)[:, jmid, :]
                plt[3][3] = Array(k_ηf)[:, jmid, :]
                plt[4][3] = Array(ηϕ)[:, jmid, :]
                # display(fig)
                save(joinpath(out_dir, "output_PW3D_$(lpad(it, 4, '0')).png"), fig)
            end
        end
        # stopping criterion — check centre column at top
        ((inv(ϕ_bg) * Array(ϕ)[nxm, nym, Int(ceil(top))]) > 1.05) && break
    end
    return
end

isdefined(Main, :_bench_sweep) || main(nx=128, nt=1, backend=:auto, verbose=true, do_plot=true)
