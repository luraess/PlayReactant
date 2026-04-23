using Reactant
using PrettyChairmarks

function main_react()
    nx = ny = 256
    nt = 100

    T1 = Reactant.ConcreteRArray(ones(nx, ny))
    T2 = Reactant.ConcreteRArray(ones(nx, ny))

    function compute!(T1, T2, nt)
        @trace for it = 1:nt
            # println("step $it")
            T2 .= T1 .* 2.0
            # T1, T2 = T2, T1
            copyto!(T1, T2)
        end
        return
    end

    compute_react! = @compile sync=true compute!(T1, T2, nt)
    compute_react!(T1, T2, nt)

    return @bs compute_react!(T1, T2, nt)
end

main_react()
