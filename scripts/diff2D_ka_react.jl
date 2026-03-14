using Reactant
using KernelAbstractions
import CUDA
using CairoMakie
using PrettyChairmarks

@kernel inbounds = true function diffusion_kernel!(T2, T, D, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    if (ix > 1 && ix < size(T, 1)) && (iy > 1 && iy < size(T, 2))
        T2[ix, iy] = T[ix, iy] + dt * D * ((T[ix+1, iy] - 2.0 * T[ix, iy] + T[ix-1, iy]) / dx / dx +
                                           (T[ix, iy+1] - 2.0 * T[ix, iy] + T[ix, iy-1]) / dy / dy)
    end
end

function diffusion_step!(T2, T, D, dt, dx, dy)
    backend = KernelAbstractions.get_backend(T)
    diffusion_kernel!(backend, 256, size(T))(T2, T, D, dt, dx, dy)
    return
end

function main_react(; plt=false)
    Lx, Ly = 10.0, 10.0
    D = 1.0

    nx = ny = 512
    nt = 100

    dx, dy = Lx / nx, Ly / ny
    dt = min(dx, dy)^2 / D / 4.1
    xc, yc = LinRange(dx / 2, Lx - dx / 2, nx), LinRange(dy / 2, Ly - dy / 2, ny)

    T = Reactant.ConcreteRArray(@. exp(-(xc - Lx / 2)^2 - (yc' - Ly / 2)^2))
    T2 = copy(T)

    function compute!(T2, T, D, dt, dx, dy, nt)
        @trace for it = 1:nt
            # println("step $it")
            diffusion_step!(T2, T, D, dt, dx, dy)
            copyto!(T, T2) # T, T2 = T2, T seems not ideal in Reactant
        end
        return
    end

    compute_react! = @compile sync = true raise = true compute!(T2, T, D, dt, dx, dy, nt)
    compute_react!(T2, T, D, dt, dx, dy, nt)

    if plt
        f = Figure()
        ax = Axis(f[1, 1], aspect=DataAspect())
        hm = heatmap!(ax, xc, yc, convert(Array, T); colormap=:turbo, colorrange=(0, 1))
        Colorbar(f[1, 2], hm)
        display(f)
    else
        println("max(T) = $(maximum(abs, convert(Array, T)))")
    end

    return @bs compute_react!(T2, T, D, dt, dx, dy, nt)
end

main_react(; plt=false)
