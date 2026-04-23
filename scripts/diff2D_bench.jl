using Reactant
using KernelAbstractions
const KA = KernelAbstractions
import CUDA
using CairoMakie
using Chairmarks

# Move a host array to the target backend
to_device(A, use_cuda::Bool) = use_cuda ? CUDA.CuArray(A) : A

# ---- Broadcast approach ----

@views Lapl(A, dx, dy) = (A[3:end, 2:end-1] .- 2.0 .* A[2:end-1, 2:end-1] .+ A[1:end-2, 2:end-1]) ./ dx^2 .+
                         (A[2:end-1, 3:end] .- 2.0 .* A[2:end-1, 2:end-1] .+ A[2:end-1, 1:end-2]) ./ dy^2

@views function diffusion_step_bc!(T, D, dt, dx, dy)
    T[2:end-1, 2:end-1] .+= dt .* D .* Lapl(T, dx, dy)
    return
end

function compute_bc!(T, D, dt, dx, dy, nt)
    @trace for it = 1:nt
        diffusion_step_bc!(T, D, dt, dx, dy)
    end
    return
end

function compute_bc_plain!(T, D, dt, dx, dy, nt)
    for it = 1:nt
        diffusion_step_bc!(T, D, dt, dx, dy)
    end
    return
end

# ---- KernelAbstractions approach ----

@kernel inbounds = true function diffusion_kernel!(T2, T, D, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    if (ix > 1 && ix < size(T, 1)) && (iy > 1 && iy < size(T, 2))
        T2[ix, iy] = T[ix, iy] + dt * D * ((T[ix+1, iy] - 2.0 * T[ix, iy] + T[ix-1, iy]) / dx / dx +
                                           (T[ix, iy+1] - 2.0 * T[ix, iy] + T[ix, iy-1]) / dy / dy)
    end
end

function compute_ka!(T2, T, D, dt, dx, dy, nt)
    backend = KA.get_backend(T)
    @trace for it = 1:nt
        diffusion_kernel!(backend, 256, size(T))(T2, T, D, dt, dx, dy)
        copyto!(T, T2)
    end
    KA.synchronize(backend)
    return
end

function compute_ka_plain!(T2, T, D, dt, dx, dy, nt)
    backend = KA.get_backend(T)
    for it = 1:nt
        diffusion_kernel!(backend, 256, size(T))(T2, T, D, dt, dx, dy)
        copyto!(T, T2)
    end
    KA.synchronize(backend)
    return
end

# ---- Setup ----

function bench(; nx=4*1024, ny=4*1024, nt=10, use_cuda::Bool=CUDA.functional())
    use_cuda ? Reactant.set_default_backend("gpu") : Reactant.set_default_backend("cpu")
    use_cuda && !CUDA.functional() && error("use_cuda=true requested but no functional CUDA device found")
    Lx, Ly = 10.0, 10.0
    D = 1.0

    dx, dy = Lx / nx, Ly / ny
    dt = min(dx, dy)^2 / D / 4.1
    xc = LinRange(dx / 2, Lx - dx / 2, nx)
    yc = LinRange(dy / 2, Ly - dy / 2, ny)

    T0 = @. exp(-(xc - Lx / 2)^2 - (yc' - Ly / 2)^2)  # host Float64 Array
    println("Backend: $(use_cuda ? "CUDA GPU" : "CPU")")

    # Plain broadcast
    T_bc_plain = to_device(copy(T0), use_cuda)
    compute_bc_plain!(T_bc_plain, D, dt, dx, dy, nt)
    println("Broadcast plain:         max(T) = $(maximum(abs, Array(T_bc_plain)))")

    # Plain KA (backend inferred from array type)
    T_ka_plain  = to_device(copy(T0), use_cuda)
    T2_ka_plain = to_device(copy(T0), use_cuda)
    compute_ka_plain!(T2_ka_plain, T_ka_plain, D, dt, dx, dy, nt)
    println("KA kernels plain:        max(T) = $(maximum(abs, Array(T_ka_plain)))")

    # Reactant broadcast
    T_bc = Reactant.ConcreteRArray(copy(T0))
    compute_bc_react! = @compile sync=true raise=true compute_bc!(T_bc, D, dt, dx, dy, nt)
    compute_bc_react!(T_bc, D, dt, dx, dy, nt)
    println("Broadcast Reactant:      max(T) = $(maximum(abs, convert(Array, T_bc)))")

    # Reactant KA
    T_ka  = Reactant.ConcreteRArray(copy(T0))
    T2_ka = Reactant.ConcreteRArray(copy(T0))
    compute_ka_react! = @compile sync=true raise=true compute_ka!(T2_ka, T_ka, D, dt, dx, dy, nt)
    compute_ka_react!(T2_ka, T_ka, D, dt, dx, dy, nt)
    println("KA kernels Reactant:     max(T) = $(maximum(abs, convert(Array, T_ka)))")

    # Benchmark
    println("\n--- Benchmark (nx=$nx, ny=$ny, nt=$nt) ---")

    print("\nBroadcast plain:         ")
    display(@b compute_bc_plain!(T_bc_plain, D, dt, dx, dy, nt))

    print("\nKA kernels plain:        ")
    display(@b compute_ka_plain!(T2_ka_plain, T_ka_plain, D, dt, dx, dy, nt))

    print("\nBroadcast Reactant:      ")
    display(@b compute_bc_react!(T_bc, D, dt, dx, dy, nt))

    print("\nKA kernels Reactant:     ")
    display(@b compute_ka_react!(T2_ka, T_ka, D, dt, dx, dy, nt))

    return
end

bench(; nx=16*1024, ny=16*1024)
