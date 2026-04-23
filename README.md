# PlayReactant
Checking out Reactant.jl

## Benchmarking

Results from running [scripts/diff2D_bench.jl](scripts/diff2D_bench.jl) on an Nvidia GH200 (Alps)

```
julia> include("scripts/diff2D_bench.jl")
Backend: CUDA GPU
Broadcast plain:         max(T) = 0.9999961793198618
KA kernels plain:        max(T) = 0.9999961793198618
Broadcast Reactant:      max(T) = 0.9999961793198618
KA kernels Reactant:     max(T) = 0.9999961793198618

--- Benchmark (nx=16384, ny=16384, nt=10) ---

Broadcast plain:         104.303 ms (1593 allocs: 116.516 KiB, 2.89% gc time)

KA kernels plain:        33.518 ms (719 allocs: 16.391 KiB)

Broadcast Reactant:      38.362 ms (18 allocs: 576 bytes)

KA kernels Reactant:     51.807 ms (24 allocs: 720 bytes)
```
