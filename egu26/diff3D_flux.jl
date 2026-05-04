using CairoMakie
using KernelAbstractions
const KA = KernelAbstractions
import CUDA
using Reactant
using Chairmarks

@kernel inbounds = true function update_q!(qx, qy, qz, H, λ, dx, dy, dz)
    ix, iy, iz = @index(Global, NTuple)
    nx, ny, nz = size(H)
    if ix < nx qx[ix+1, iy,   iz  ] = -λ * (H[ix+1, iy, iz] - H[ix, iy, iz]) / dx end
    if iy < ny qy[ix,   iy+1, iz  ] = -λ * (H[ix, iy+1, iz] - H[ix, iy, iz]) / dy end
    if iz < nz qz[ix,   iy,   iz+1] = -λ * (H[ix, iy, iz+1] - H[ix, iy, iz]) / dz end
end

@kernel inbounds = true function update_H!(H, qx, qy, qz, dt, dx, dy, dz)
    ix, iy, iz = @index(Global, NTuple)
    H[ix, iy, iz] -= dt * ((qx[ix+1, iy, iz] - qx[ix, iy, iz]) / dx +
                           (qy[ix, iy+1, iz] - qy[ix, iy, iz]) / dy +
                           (qz[ix, iy, iz+1] - qz[ix, iy, iz]) / dz)
end

function diffusion_step!(backend, H, qx, qy, qz, λ, dt, dx, dy, dz)
    nx, ny, nz = size(H)
    update_q!(backend, (256), (nx, ny, nz))(qx, qy, qz, H, λ, dx, dy, dz)
    update_H!(backend, (256), (nx, ny, nz))(H, qx, qy, qz, dt, dx, dy, dz)
    return
end

function time_loop!(H, qx, qy, qz, λ, dt, dx, dy, dz, nt)
    backend = KA.get_backend(H)
    for it in 1:nt
        diffusion_step!(backend, H, qx, qy, qz, λ, dt, dx, dy, dz)
    end
    KA.synchronize(backend)
    return
end

function time_loop_react!(H, qx, qy, qz, λ, dt, dx, dy, dz, nt)
    backend = KA.get_backend(H)
    @trace for it in 1:nt
        diffusion_step!(backend, H, qx, qy, qz, λ, dt, dx, dy, dz)
    end
    KA.synchronize(backend)
    return
end

function time_loop_react_bcast!(H, qx, qy, qz, λ, dt, dx, dy, dz, nt)
    @trace for it in 1:nt
        qx[2:end-1, :, :] .= .-λ .* (H[2:end, :, :] .- H[1:end-1, :, :]) ./ dx
        qy[:, 2:end-1, :] .= .-λ .* (H[:, 2:end, :] .- H[:, 1:end-1, :]) ./ dy
        qz[:, :, 2:end-1] .= .-λ .* (H[:, :, 2:end] .- H[:, :, 1:end-1]) ./ dz
        H .-= dt .* ((qx[2:end, :, :] .- qx[1:end-1, :, :]) ./ dx .+
                     (qy[:, 2:end, :] .- qy[:, 1:end-1, :]) ./ dy .+
                     (qz[:, :, 2:end] .- qz[:, :, 1:end-1]) ./ dz)
    end
    return
end

to_device(A, use_cuda::Bool) = use_cuda ? CUDA.CuArray(A) : A

function runme(; nx=32, ny=32, nz=32, nt=10, dtype=Float64, use_cuda::Bool=CUDA.functional(), do_plot::Bool=false)
    use_cuda && !CUDA.functional() && error("use_cuda=true requested but no functional CUDA device found")
    use_cuda ? Reactant.set_default_backend("gpu") : Reactant.set_default_backend("cpu")
    backend = use_cuda ? CUDA.CUDABackend() : CPU()

    lx, ly, lz = 10.0, 10.0, 10.0
    λ = 1.0
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    dt = min(dx, dy, dz)^2 / λ / 6.1

    coord = (x=LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx),
             y=LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny),
             z=LinRange(-lz / 2 + dz / 2, lz / 2 - dz / 2, nz),)

    H0 = @. exp(-coord.x^2 - coord.y'^2 - $(reshape(coord.z, 1, 1, nz))^2)
    println("Backend: $(use_cuda ? "CUDA GPU" : "CPU")")

    # --- plain KA ---
    H_ka  = to_device(convert(Array{dtype}, copy(H0)), use_cuda)
    qx_ka = KA.zeros(backend, dtype, nx + 1, ny, nz)
    qy_ka = KA.zeros(backend, dtype, nx, ny + 1, nz)
    qz_ka = KA.zeros(backend, dtype, nx, ny, nz + 1)

    time_loop!(H_ka, qx_ka, qy_ka, qz_ka, λ, dt, dx, dy, dz, nt)
    println("KA plain:       max(H) = $(maximum(abs, Array(H_ka)))")
    P_KA = Array(H_ka)

    bm_ka = @b time_loop!(H_ka, qx_ka, qy_ka, qz_ka, λ, dt, dx, dy, dz, nt)

    # --- Reactant ---
    H_r  = Reactant.ConcreteRArray(convert(Array{dtype}, copy(H0)))
    qx_r = Reactant.ConcreteRArray(zeros(dtype, nx + 1, ny, nz))
    qy_r = Reactant.ConcreteRArray(zeros(dtype, nx, ny + 1, nz))
    qz_r = Reactant.ConcreteRArray(zeros(dtype, nx, ny, nz + 1))

    compute_react! = @compile sync=true raise=true time_loop_react!(H_r, qx_r, qy_r, qz_r, λ, dt, dx, dy, dz, nt)
    compute_react!(H_r, qx_r, qy_r, qz_r, λ, dt, dx, dy, dz, nt)
    println("Reactant:       max(H) = $(maximum(abs, convert(Array, H_r)))")
    P_re = convert(Array, H_r)

    bm_r = @b compute_react!(H_r, qx_r, qy_r, qz_r, λ, dt, dx, dy, dz, nt)

    # --- Reactant broadcast ---
    H_rb  = Reactant.ConcreteRArray(convert(Array{dtype}, copy(H0)))
    qx_rb = Reactant.ConcreteRArray(zeros(dtype, nx + 1, ny, nz))
    qy_rb = Reactant.ConcreteRArray(zeros(dtype, nx, ny + 1, nz))
    qz_rb = Reactant.ConcreteRArray(zeros(dtype, nx, ny, nz + 1))

    compute_react_bcast! = @compile sync=true time_loop_react_bcast!(H_rb, qx_rb, qy_rb, qz_rb, λ, dt, dx, dy, dz, nt)
    compute_react_bcast!(H_rb, qx_rb, qy_rb, qz_rb, λ, dt, dx, dy, dz, nt)
    println("Reactant bcast: max(H) = $(maximum(abs, convert(Array, H_rb)))")
    P_rb = convert(Array, H_rb)

    bm_rb = @b compute_react_bcast!(H_rb, qx_rb, qy_rb, qz_rb, λ, dt, dx, dy, dz, nt)

    # --- plot (xy mid-plane slice) ---
    if do_plot
        iz_mid = nz ÷ 2
        fig = Figure(size=(300, 700))
        ax1 = Axis(fig[1, 1]; title="KA plain",       aspect=DataAspect())
        ax2 = Axis(fig[2, 1]; title="Reactant KA",    aspect=DataAspect())
        ax3 = Axis(fig[3, 1]; title="Reactant bcast", aspect=DataAspect())
        hm1 = heatmap!(ax1, coord.x, coord.y, P_KA[:, :, iz_mid]; colorrange=(0, .8))
        hm2 = heatmap!(ax2, coord.x, coord.y, P_re[:, :, iz_mid]; colorrange=(0, .8))
        hm3 = heatmap!(ax3, coord.x, coord.y, P_rb[:, :, iz_mid]; colorrange=(0, .8))
        Colorbar(fig[1, 2], hm1)
        Colorbar(fig[2, 2], hm2)
        Colorbar(fig[3, 2], hm3)
        save("output3D.png", fig)
    end

    # --- report ---
    A_eff = 2 * (sizeof(H_ka) + sizeof(qx_ka) + sizeof(qy_ka) + sizeof(qz_ka)) * 1e-9 * nt
    println("\n--- Benchmark (nx=$nx, ny=$ny, nz=$nz, nt=$nt) ---")
    println("KA plain       time loop: Teff = $(round(A_eff / bm_ka.time,  digits=2)) GB/s")
    println("Reactant KA    time loop: Teff = $(round(A_eff / bm_r.time,   digits=2)) GB/s")
    println("Reactant bcast time loop: Teff = $(round(A_eff / bm_rb.time,  digits=2)) GB/s")

    return
end

res = 1024
# runme(; nx=res, ny=res, nz=res, use_cuda=false)
runme(; nx=res, ny=res, nz=res, nt=10, use_cuda=true)
