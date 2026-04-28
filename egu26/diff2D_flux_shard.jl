using Reactant
using KernelAbstractions
const KA = KernelAbstractions
using CairoMakie
using Chairmarks
import CUDA
using Preferences, UUIDs

# Use IFRT runtime (required for multi-device / sharding)
Reactant.set_default_backend("gpu")
Preferences.set_preferences!(UUID("3c362404-f566-11ee-1572-e11a4b42c853"), "xla_runtime" => "IFRT")

# sizeof is unreliable for sharded ConcreteRArrays — use length × element size instead
nbytes(A) = length(A) * sizeof(eltype(A))

# Mesh helper — returns (D, D) for a square mesh, errors otherwise
function mesh_factors(N::Int)
    D = isqrt(N)
    D^2 == N || throw(ArgumentError("Number of devices N=$N is not a perfect square; cannot form a square mesh"))
    return D, D
end

# KA kernels (flux-based)
@kernel inbounds = true function update_q!(qx, qy, H, λ, dx, dy)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(H)
    if ix < nx qx[ix+1, iy  ] = -λ * (H[ix+1, iy] - H[ix, iy]) / dx end
    if iy < ny qy[ix,   iy+1] = -λ * (H[ix, iy+1] - H[ix, iy]) / dy end
end

@kernel inbounds = true function update_H!(H, qx, qy, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    H[ix, iy] -= dt * ((qx[ix+1, iy] - qx[ix, iy]) / dx +
                       (qy[ix, iy+1] - qy[ix, iy]) / dy)
end

function diffusion_step!(backend, H, qx, qy, λ, dt, dx, dy)
    nx, ny = size(H)
    update_q!(backend, 256, (nx, ny))(qx, qy, H, λ, dx, dy)
    update_H!(backend, 256, (nx, ny))(H, qx, qy, dt, dx, dy)
    return
end

function time_loop_react!(H, qx, qy, λ, dt, dx, dy, nt)
    backend = KA.get_backend(H)
    @trace for _ in 1:nt
        diffusion_step!(backend, H, qx, qy, λ, dt, dx, dy)
    end
    KA.synchronize(backend)
    return
end

function time_loop_react_bcast!(H, qx, qy, λ, dt, dx, dy, nt)
    @trace for _ in 1:nt
        qx[2:end-1, :] .= .-λ .* (H[2:end, :] .- H[1:end-1, :]) ./ dx
        qy[:, 2:end-1] .= .-λ .* (H[:, 2:end] .- H[:, 1:end-1]) ./ dy
        H .-= dt .* ((qx[2:end, :] .- qx[1:end-1, :]) ./ dx .+
                     (qy[:, 2:end] .- qy[:, 1:end-1]) ./ dy)
    end
    return
end

# Main — nx, ny are the LOCAL (per-device) grid sizes;
#        the global grid is (nx*Dx) × (ny*Dy) and is sharded across Dx×Dy devices.
function runme(; nx=512, ny=512, nt=10, dtype=Float64, do_plot::Bool=false, ndev::Union{Int,Nothing}=nothing)
    !CUDA.functional() && error("No functional CUDA device found")
    # Reactant.set_default_backend("gpu")
    println("Backend: CUDA GPU")

    Ndev = isnothing(ndev) ? length(Reactant.devices()) : ndev
    Dx, Dy = mesh_factors(Ndev)
    @info "Devices: $Ndev  →  mesh $(Dx)×$(Dy)"

    # Global grid dimensions
    Nx, Ny = nx * Dx, ny * Dy
    @info "Local grid: $(nx)×$(ny)  |  Global grid: $(Nx)×$(Ny)"

    lx, ly = 10.0, 10.0
    λ      = 1.0
    dx, dy = lx / Nx, ly / Ny
    dt     = min(dx, dy)^2 / λ / 4.1

    coord = (x = LinRange(-lx/2 + dx/2, lx/2 - dx/2, Nx),
             y = LinRange(-ly/2 + dy/2, ly/2 - dy/2, Ny),)

    H0 = @. exp(-coord.x^2 - coord.y'^2)

    # ---- Sharding setup ----
    axis    = (:x, :y)
    devices = reshape(Reactant.devices()[1:Ndev], Dx, Dy)
    mesh    = Sharding.Mesh(devices, axis)
    shard   = Sharding.NamedSharding(mesh, axis)

    # ---- Sharded Reactant KA ----
    # H_r  = Reactant.ConcreteRArray(convert(Array{dtype}, copy(H0)); sharding=shard)
    # qx_r = Reactant.ConcreteRArray(zeros(dtype, Nx + 1, Ny); sharding=shard)
    # qy_r = Reactant.ConcreteRArray(zeros(dtype, Nx, Ny + 1); sharding=shard)

    # compute_react! = @compile sync=true raise=true time_loop_react!(H_r, qx_r, qy_r, λ, dt, dx, dy, nt)
    # compute_react!(H_r, qx_r, qy_r, λ, dt, dx, dy, nt)
    # println("Reactant KA    (sharded $Ndev GPUs): max(H) = $(maximum(abs, convert(Array, H_r)))")
    # do_plot && (P_r = convert(Array, H_r))

    # bm_r = @b compute_react!(H_r, qx_r, qy_r, λ, dt, dx, dy, nt)

    # ---- Sharded Reactant broadcast ----
    H_rb  = Reactant.ConcreteRArray(convert(Array{dtype}, copy(H0));  sharding=shard)
    qx_rb = Reactant.ConcreteRArray(zeros(dtype, Nx + 1, Ny); sharding=shard)
    qy_rb = Reactant.ConcreteRArray(zeros(dtype, Nx, Ny + 1); sharding=shard)

    compute_react_bcast! = @compile sync=true time_loop_react_bcast!(H_rb, qx_rb, qy_rb, λ, dt, dx, dy, nt)
    compute_react_bcast!(H_rb, qx_rb, qy_rb, λ, dt, dx, dy, nt)
    println("Reactant bcast (sharded $Ndev GPUs): max(H) = $(maximum(abs, convert(Array, H_rb)))")
    do_plot && (P_rb = convert(Array, H_rb))

    bm_rb = @b compute_react_bcast!(H_rb, qx_rb, qy_rb, λ, dt, dx, dy, nt)

    # ---- Plot ----
    if do_plot
        fig = Figure(size=(300, 500))
        ax1 = Axis(fig[1, 1]; title="Reactant KA (sharded)",    aspect=DataAspect())
        ax2 = Axis(fig[2, 1]; title="Reactant bcast (sharded)", aspect=DataAspect())
        # hm1 = heatmap!(ax1, coord.x, coord.y, P_r;  colorrange=(0, .8))
        hm2 = heatmap!(ax2, coord.x, coord.y, P_rb; colorrange=(0, .8))
        # Colorbar(fig[1, 2], hm1)
        Colorbar(fig[2, 2], hm2)
        save("output_shard.png", fig)
    end
    A_eff = 2 * (nbytes(H_rb) + nbytes(qx_rb) + nbytes(qy_rb)) * 1e-9 * nt / Ndev
    println("\n--- Benchmark (local=$(nx)×$(ny), global=$(Nx)×$(Ny), nt=$nt, Ndev=$Ndev) ---")
    # println("Reactant KA    ($Ndev GPU$(Ndev > 1 ? "s" : "")): Teff = $(round(A_eff / bm_r.time,  digits=2)) GB/s  |  $bm_r")
    println("Reactant bcast ($Ndev GPU$(Ndev > 1 ? "s" : "")): Teff = $(round(A_eff / bm_rb.time, digits=2)) GB/s  |  $bm_rb")

    return
end

res = 16 * 1024
runme(; nx=res, ny=res, nt=10, do_plot=false)
