using Reactant
using KernelAbstractions
import CUDA
using CairoMakie
using PrettyChairmarks

@kernel inbounds=true function diffusion_kernel!(T2, T, D, dt, dx)
    i = @index(Global)
    if i > 1 && i < length(T)
        T2[i] = T[i] + dt * D * (T[i+1] - 2.0 * T[i] + T[i-1]) / dx^2
    end
end

function diffusion_step!(T2, T, D, dt, dx)
    backend = KernelAbstractions.get_backend(T)
    diffusion_kernel!(backend, 32, length(T))(T2, T, D, dt, dx)
    return
end

function main_react()
    L = 10.0
    D = 1.0

    nx = 512
    nt = 1000

    dx = L / nx
    dt = dx^2 / D / 2.1
    xc = LinRange(dx / 2, L - dx / 2, nx)

    T = Reactant.ConcreteRArray(@. exp(-(xc - L / 2)^2))
    T2 = copy(T)
    T0 = copy(T)

    function compute!(T2, T, D, dt, dx, nt)
        @trace for it = 1:nt
            # println("step $it")
            diffusion_step!(T2, T, D, dt, dx)
            copyto!(T, T2) # T, T2 = T2, T seems not ideal in Reactant
        end
        return
    end

    compute_react! = @compile sync=true raise=true compute!(T2, T, D, dt, dx, nt)
    compute_react!(T2, T, D, dt, dx, nt)

    f = Figure()
    ax = Axis(f[1, 1])
    lines!(ax, xc, convert(Array, T0))
    lines!(ax, xc, convert(Array, T))
    display(f)

    return @bs compute_react!(T2, T, D, dt, dx, nt)
end

main_react()
