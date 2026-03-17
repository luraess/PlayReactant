using Reactant
using CairoMakie
using PrettyChairmarks
using Preferences, UUIDs

Reactant.set_default_backend("gpu")
Preferences.set_preferences!(UUID("3c362404-f566-11ee-1572-e11a4b42c853"), "xla_runtime" => "IFRT")

@views Lapl(A, dx, dy) = (A[3:end, 2:end-1] .- 2.0 .* A[2:end-1, 2:end-1] .+ A[1:end-2, 2:end-1]) ./ dx^2 .+
                         (A[2:end-1, 3:end] .- 2.0 .* A[2:end-1, 2:end-1] .+ A[2:end-1, 1:end-2]) ./ dy^2

@views function diffusion_step!(T, D, dt, dx, dy)
    T[2:end-1, 2:end-1] .+= dt .* D .* Lapl(T, dx, dy)
    return
end

function main_react(; plt=false)
    Lx, Ly = 10.0, 10.0
    D  = 1.0

    nx = ny = 2048
    nt = 100

    axis = (:x, :y)
    mesh = Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), axis)

    dx, dy = Lx / nx, Ly / ny
    dt = min(dx, dy)^2 / D / 4.1
    xc, yc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)

    # T  = Reactant.ConcreteRArray(@. exp(-(xc - Lx/2)^2 - (yc' - Ly/2)^2))
    T  = Reactant.ConcreteRArray(@. exp(-(xc - Lx/2)^2 - (yc' - Ly/2)^2); sharding=Sharding.NamedSharding(mesh, axis))

    function compute!(T, D, dt, dx, dy, nt)
        @trace for it = 1:nt
            # println("step $it")
            diffusion_step!(T, D, dt, dx, dy)
        end
        return
    end

    compute_react! = @compile sync=true compute!(T, D, dt, dx, dy, nt)
    compute_react!(T, D, dt, dx, dy, nt)

    if plt
        f = Figure()
        ax = Axis(f[1, 1], aspect=DataAspect())
        hm = heatmap!(ax, xc, yc, convert(Array, T); colormap=:turbo, colorrange=(0, 1))
        Colorbar(f[1, 2], hm)
        display(f)
    else
        println("max(T) = $(maximum(abs, convert(Array, T)))")
    end

    return @bs compute_react!(T, D, dt, dx, dy, nt)
end

main_react(; plt=false)
