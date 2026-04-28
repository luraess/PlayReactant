using CairoMakie
using KernelAbstractions
const KA = KernelAbstractions
import CUDA
using Reactant
using Chairmarks

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

function time_loop!(H, qx, qy, λ, dt, dx, dy, nt)
    backend = KA.get_backend(H)
    for it in 1:nt
        diffusion_step!(backend, H, qx, qy, λ, dt, dx, dy)
    end
    KA.synchronize(backend)
    return
end

function time_loop_react!(H, qx, qy, λ, dt, dx, dy, nt)
    backend = KA.get_backend(H)
    @trace for it in 1:nt
        diffusion_step!(backend, H, qx, qy, λ, dt, dx, dy)
    end
    KA.synchronize(backend)
    return
end

function time_loop_react_bcast!(H, qx, qy, λ, dt, dx, dy, nt)
    @trace for it in 1:nt
        qx[2:end-1, :] .= .-λ .* (H[2:end, :] .- H[1:end-1, :]) ./ dx
        qy[:, 2:end-1] .= .-λ .* (H[:, 2:end] .- H[:, 1:end-1]) ./ dy
        H .-= dt .* ((qx[2:end, :] .- qx[1:end-1, :]) ./ dx .+
                     (qy[:, 2:end] .- qy[:, 1:end-1]) ./ dy)
    end
    return
end

to_device(A, use_cuda::Bool) = use_cuda ? CUDA.CuArray(A) : A

function runme(; nx=64, ny=64, nt=50, dtype=Float64, use_cuda::Bool=CUDA.functional())
    use_cuda && !CUDA.functional() && error("use_cuda=true requested but no functional CUDA device found")
    use_cuda ? Reactant.set_default_backend("gpu") : Reactant.set_default_backend("cpu")
    backend = use_cuda ? CUDA.CUDABackend() : CPU()

    lx, ly = 10.0, 10.0
    λ = 1.0
    dx, dy = lx / nx, ly / ny
    dt = min(dx, dy)^2 / λ / 4.1

    coord = (x=LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx),
             y=LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny),)

    H0 = @. exp(-coord.x^2 - coord.y'^2)
    println("Backend: $(use_cuda ? "CUDA GPU" : "CPU")")

    # --- plain KA ---
    H_ka  = to_device(convert(Array{dtype}, copy(H0)), use_cuda)
    qx_ka = KA.zeros(backend, dtype, nx + 1, ny)
    qy_ka = KA.zeros(backend, dtype, nx, ny + 1)

    time_loop!(H_ka, qx_ka, qy_ka, λ, dt, dx, dy, nt)
    println("KA plain:       max(H) = $(maximum(abs, Array(H_ka)))")
    P_KA = Array(H_ka)

    bm_ka = @b time_loop!(H_ka, qx_ka, qy_ka, λ, dt, dx, dy, nt)

    # --- Reactant ---
    H_r  = Reactant.ConcreteRArray(convert(Array{dtype}, copy(H0)))
    qx_r = Reactant.ConcreteRArray(zeros(dtype, nx + 1, ny))
    qy_r = Reactant.ConcreteRArray(zeros(dtype, nx, ny + 1))

    compute_react! = @compile sync=true raise=true time_loop_react!(H_r, qx_r, qy_r, λ, dt, dx, dy, nt)
    compute_react!(H_r, qx_r, qy_r, λ, dt, dx, dy, nt)
    println("Reactant:       max(H) = $(maximum(abs, convert(Array, H_r)))")
    P_re = convert(Array, H_r)

    bm_r = @b compute_react!(H_r, qx_r, qy_r, λ, dt, dx, dy, nt)

    # --- Reactant broadcast ---
    H_rb  = Reactant.ConcreteRArray(convert(Array{dtype}, copy(H0)))
    qx_rb = Reactant.ConcreteRArray(zeros(dtype, nx + 1, ny))
    qy_rb = Reactant.ConcreteRArray(zeros(dtype, nx, ny + 1))

    compute_react_bcast! = @compile sync=true raise=true time_loop_react_bcast!(H_rb, qx_rb, qy_rb, λ, dt, dx, dy, nt)
    compute_react_bcast!(H_rb, qx_rb, qy_rb, λ, dt, dx, dy, nt)
    println("Reactant bcast: max(H) = $(maximum(abs, convert(Array, H_rb)))")
    P_rb = convert(Array, H_rb)

    bm_rb = @b compute_react_bcast!(H_rb, qx_rb, qy_rb, λ, dt, dx, dy, nt)

    # --- plot ---
    fig = Figure(size=(300, 700))
    ax1 = Axis(fig[1, 1]; title="KA plain",       aspect=DataAspect())
    ax2 = Axis(fig[2, 1]; title="Reactant KA",    aspect=DataAspect())
    ax3 = Axis(fig[3, 1]; title="Reactant bcast", aspect=DataAspect())
    hm1 = heatmap!(ax1, coord.x, coord.y, P_KA; colorrange=(0, .8))
    hm2 = heatmap!(ax2, coord.x, coord.y, P_re;  colorrange=(0, .8))
    hm3 = heatmap!(ax3, coord.x, coord.y, P_rb;  colorrange=(0, .8))
    Colorbar(fig[1, 2], hm1)
    Colorbar(fig[2, 2], hm2)
    Colorbar(fig[3, 2], hm3)
    display(fig)

    # --- report ---
    A_eff = 2 * (sizeof(H_ka) + sizeof(qx_ka) + sizeof(qy_ka)) * 1e-9
    println("\n--- Benchmark (nx=$nx, ny=$ny, nt=$nt) ---")
    println("KA plain       time loop: Teff = $(round(A_eff / bm_ka.time,  digits=2)) GB/s")
    println("Reactant KA    time loop: Teff = $(round(A_eff / bm_r.time,   digits=2)) GB/s")
    println("Reactant bcast time loop: Teff = $(round(A_eff / bm_rb.time,  digits=2)) GB/s")

    return
end

res = 1 * 64
runme(; nx=res, ny=res, use_cuda=false)
# runme(; nx=res, ny=res, use_cuda=true)
