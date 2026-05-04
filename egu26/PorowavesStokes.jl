using LinearAlgebra, Statistics, Printf
using CairoMakie

# helper functions
@views inn(A)  = A[2:end-1, 2:end-1]
@views avx(A)  = @. 0.5 * (A[1:end-1, :] + A[2:end, :])
@views avy(A)  = @. 0.5 * (A[:, 1:end-1] + A[:, 2:end])
@views avxi(A) = @. 0.5 * (A[1:end-1, 2:end-1] + A[2:end, 2:end-1])
@views avyi(A) = @. 0.5 * (A[2:end-1, 1:end-1] + A[2:end-1, 2:end])
@views av(A)   = @. 0.25 * (A[1:end-1, 1:end-1] + A[1:end-1, 2:end] + A[2:end, 1:end-1] + A[2:end, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

@views function neumann_bcs_ap!(A)
    A[1, :]   .= A[2, :]
    A[end, :] .= A[end-1, :]
    A[:, 1]   .= A[:, 2]
    A[:, end] .= A[:, end-1]
end

@views function main()
    # independent physics
    lc     = 1.0
    Δρg    = 1.0
    ηϕ_bg  = 1.0
    # scales
    psc    = Δρg * lc
    tsc    = ηϕ_bg / psc
    # nondim numbers
    R      = 100.0 # 100.0
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
    nx      = 40  # grid size
    ny      = ceil(Int, nx * ra)
    tol     = 1e-3
    cfl_dt  = 2e-2
    ϕ_rel   = 1e-1
    nt      = 1 #50 #1e5
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
    # dmp solid
    re_m    = 2.3π
    r       = 0.5
    lτ_re_m = min(lx, ly) / re_m
    vdτ     = min(dx, dy) / sqrt(4.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    νdτ     = vdτ * lτ_re_m
    dτ_Pr   = r / θ_dτ
    # dmp fluid
    cfl     = 1 / sqrt(2.1)
    # initialisation
    ϕs      = zeros(nx, ny)
    ηϕ      = zeros(nx, ny)
    Pe      = zeros(nx, ny)
    Vxs     = zeros(nx - 1, ny)
    Vys     = zeros(nx, ny - 1)
    τxx     = zeros(nx, ny)
    τyy     = zeros(nx, ny)
    τxy     = zeros(nx - 1, ny - 1)
    ∇Vs     = zeros(nx - 2, ny - 2)
    dϕdt    = zeros(nx - 2, ny - 2)
    qDx     = zeros(nx - 1, ny - 2)
    qDy     = zeros(nx - 2, ny - 1)
    Vxf     = zeros(nx - 1, ny - 2)
    Vyf     = zeros(nx - 2, ny - 1)
    Qfx     = zeros(nx - 1, ny - 2)
    Qfy     = zeros(nx - 2, ny - 1)
    RVx     = zeros(nx - 1, ny - 2)
    RVy     = zeros(nx - 2, ny - 1)
    RPt     = zeros(nx - 2, ny - 2)
    RPf     = zeros(nx - 2, ny - 2)
    Rτxx    = zeros(nx, ny)
    Rτyy    = zeros(nx, ny)
    Rτxy    = zeros(nx - 1, ny - 1)
    RqDx    = zeros(nx - 1, ny - 2)
    RqDy    = zeros(nx - 2, ny - 1)
    ϕ_old   = zeros(nx, ny)
    ρt_old  = zeros(nx, ny)
    lc_loc  = zeros(nx, ny)
    re      = zeros(nx, ny)
    θ_dτ    = zeros(nx, ny)
    dτ_βf   = zeros(nx - 2, ny - 2)
    # initial conditions
    ϕ       = ϕ_bg .* (1 .+ ϕ_A .* exp.(-(xc ./ w) .^ 2 .- ((yc' .+ ly / 4) ./ w) .^ 2))
    k_ηf    = k_ηf0 .* (ϕ ./ ϕ_bg) .^ npow
    ρt      = (1.0 .- ϕ) .* ρs .+ ϕ .* ρf
    ρt_ref  = (1.0 .- ϕ_bg) .* ρs .+ ϕ_bg .* ρf
    Pt      = -ρt_ref .* (0.0 .* xc .+ yc')
    Pf      = copy(Pt)
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
    display(fig)

    nxm = (nx + 1) ÷ 2
    time_evo = [0.0]
    ϕmax_evo = [maximum(ϕ)]
    # time loop
    for it = 1:nt
        println("> step $it")
        ϕ_old  .= ϕ
        ρt_old .= (1.0 .- ϕ) .* ρs .+ ϕ .* ρf
        iter  = 0; error = 10tol
        # pt loop
        while (error > tol && iter <= maxiter) || iter < miniter
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
            Rτxy                   .=     -τxy  .+        ηs .* (diff(Vxs, dims=2) ./ dy .+ diff(Vys, dims=1) ./ dx)
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
            # error
            if mod(iter, nout) == 0
                errs = [maximum(abs, RVx), maximum(abs, RVy), maximum(abs, RPt), maximum(abs, RPf)]
                error = maximum(errs)
                @printf "  iter/ny=%d, errs: RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e \n" ceil(iter/ny) errs...
            end
        end
        # diagnostic monitoring
        push!(time_evo, time_evo[end] + dt)
        push!(ϕmax_evo, maximum(ϕ))
        # time step
        dt = cfl_dt * maximum(ϕ) / maximum(abs, dϕdt)
        # visualisation
        if mod(it, nviz) == 0 || it == 1
            plt[1][3] = Array(ϕ)
            plt[2][3] = Array(Pf .- Pt)
            plt[3][3] = Array(k_ηf)
            plt[4][3] = Array(ηϕ)
            display(fig)
        end
        # stopping criterion
        ((inv(ϕ_bg) * ϕ[nxm, Int(ceil(top))]) > 1.05) && break
    end
end

main()
