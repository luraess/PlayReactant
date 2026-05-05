# Shared Reactant backend helpers — included by all Reactant physics scripts.
# Do not use standalone; always `include` from within a script that has already
# loaded `using Reactant`.

const _CUDA_functional = try
    using CUDA
    CUDA.functional()
catch
    false
end

# backend init
function init_backend(backend::Symbol=:auto)
    # backend: :auto (detect TPU → CUDA → CPU), :gpu, :cuda, :rocm, :tpu, :cpu, :none
    resolved = if backend === :auto
        if Reactant.Accelerators.TPU.has_tpu()
            :tpu
        elseif _CUDA_functional
            :gpu
        else
            :cpu
        end
    else
        backend
    end
    backend_str = Dict(:gpu  => "gpu",
                       :cuda => "cuda",
                       :rocm => "rocm",
                       :tpu  => "tpu",
                       :cpu  => "cpu")
    if resolved !== :none
        Reactant.set_default_backend(backend_str[resolved])
        @info "Reactant backend: $resolved"
        return resolved, Reactant.ConcreteRArray, Reactant.ConcreteRNumber
    else
        @info "Plain Julia backend (no Reactant)"
        return resolved, identity, identity
    end
end

to_scalar(x::Union{AbstractFloat, Integer}) = x
to_scalar(x) = Reactant.to_number(x)
