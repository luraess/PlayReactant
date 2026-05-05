# helper functions
avx(A, ix, iy) = 0.5 * (A[ix, iy] + A[ix+1, iy])
avy(A, ix, iy) = 0.5 * (A[ix, iy] + A[ix, iy+1])
∂x(A, ix, iy, _dx) = (A[ix+1, iy] - A[ix, iy]) * _dx
∂y(A, ix, iy, _dy) = (A[ix, iy+1] - A[ix, iy]) * _dy
∂x(ϕ, V, ix, iy, _dx) = ((1.0 - 0.5 * (ϕ[ix+1, iy] + ϕ[ix+2, iy])) * V[ix+1, iy] -
                         (1.0 - 0.5 * (ϕ[ix  , iy] + ϕ[ix+1, iy])) * V[ix  , iy]) * _dx
∂y(ϕ, V, ix, iy, _dy) = ((1.0 - 0.5 * (ϕ[ix, iy+1] + ϕ[ix, iy+2])) * V[ix, iy+1] -
                         (1.0 - 0.5 * (ϕ[ix  , iy] + ϕ[ix, iy+1])) * V[ix  , iy]) * _dy

@views inn(A)  = A[2:end-1, 2:end-1]
@views avx(A)  = @. 0.5 * (A[1:end-1, :] + A[2:end, :])
@views avy(A)  = @. 0.5 * (A[:, 1:end-1] + A[:, 2:end])

macro isin(A) esc(:(checkbounds(Bool, $A, ix, iy))) end

# physics
ϕs(ϕ, ix, iy) = 1.0 - ϕ[ix, iy]

# bcs
@kernel inbounds = false function cp_x1!(A)
    _, iy = @index(Global, NTuple)
    A[1, iy] = A[2, iy]
end

@kernel inbounds = false function cp_x2!(A)
    _, iy = @index(Global, NTuple)
    A[2, iy] = A[1, iy]
end

@kernel inbounds = false function cp_y1!(A)
    ix, _ = @index(Global, NTuple)
    A[ix, 1] = A[ix, 2]
end

@kernel inbounds = false function cp_y2!(A)
    ix, _ = @index(Global, NTuple)
    A[ix, 2] = A[ix, 1]
end

get_x1(A) = @view A[1:2, :]
get_x2(A) = @view A[end-1:end, :]
get_y1(A) = @view A[:, 1:2]
get_y2(A) = @view A[:, end-1:end]

function neumann_bcs!(A)
    backend = get_backend(A)
    nx, ny = size(A)
    cp_x1!(backend, 256, (1, ny))(get_x1(A))
    cp_x2!(backend, 256, (1, ny))(get_x2(A))
    cp_y1!(backend, 256, (nx, 1))(get_y1(A))
    cp_y2!(backend, 256, (nx, 1))(get_y2(A))
end
