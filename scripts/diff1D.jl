using CairoMakie
using PrettyChairmarks

@views function diffusion_step!(T, D, dt, dx)
    T[2:end-1] .+= dt .* D .* (T[3:end] .- 2.0 .* T[2:end-1] .+ T[1:end-2]) ./ dx^2
    return
end

function main()
    L = 10.0
    D = 1.0

    nx = 512
    nt = 1000

    dx = L / nx
    dt = dx^2 / D / 2.1
    xc = LinRange(dx/2, L-dx/2, nx)

    T  = @. exp(-(xc - L/2)^2)
    T0 = copy(T)

    function compute!(T, D, dt, dx, nt)
        for it = 1:nt
            # println("step $it")
            diffusion_step!(T, D, dt, dx)
        end
    end

    compute!(T, D, dt, dx, nt)

    f = Figure()
    ax = Axis(f[1, 1])
    lines!(ax, xc, T0)
    lines!(ax, xc, T)
    display(f)

    return @bs compute!(T, D, dt, dx, nt)
end

main()
