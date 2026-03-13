using Reactant
using CairoMakie
using PrettyChairmarks

@views Lapl(A) = A[3:end] .- 2.0 .* A[2:end-1] .+ A[1:end-2]

@views function diffusion_step!(T, D, dt, dx)
    T[2:end-1] .+= dt .* D .* Lapl(T) ./ dx^2
    return
end

function main_react()
    L = 10.0
    D = 1.0

    nx = 512
    nt = 1000

    dx = L / nx
    dt = dx^2 / D / 2.1
    xc = LinRange(dx/2, L-dx/2, nx)

    T  = Reactant.ConcreteRArray(@. exp(-(xc - L/2)^2))
    T0 = copy(T)

    function compute!(T, D, dt, dx, nt)
        @trace for it = 1:nt
            # println("step $it")
            diffusion_step!(T, D, dt, dx)
        end
        return
    end

    compute_react! = @compile sync=true compute!(T, D, dt, dx, nt)
    compute_react!(T, D, dt, dx, nt)

    f = Figure()
    ax = Axis(f[1, 1])
    lines!(ax, xc, convert(Array, T0))
    lines!(ax, xc, convert(Array, T))
    display(f)

    return @bs compute_react!(T, D, dt, dx, nt)
end

main_react()
