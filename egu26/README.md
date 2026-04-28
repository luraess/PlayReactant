# 2D Diffusion with Reactant.jl

Benchmarks and correctness checks for [`diff2D_flux.jl`](diff2D_flux.jl), running three flavours of 2D diffusion on a Grace CPU and an H100 GPU:
- **KA plain** — straightforward KernelAbstractions.jl kernels
- **Reactant KA** — KA kernels compiled through Reactant.jl (`@compile`)
- **Reactant bcast** — broadcast-style stencil compiled through Reactant.jl

## Correctness issue

The middle panel (Reactant KA) produces wrong values and an asymmetric diffusion pattern, visible in the heatmap comparison below.

```julia
res = 64
runme(; nx=res, ny=res, nt=50, use_cuda=true)
```

<img src="output.png" width="300" alt="2D diffusion output — KA plain (top), Reactant KA (middle), Reactant bcast (bottom)"/>

> **Note:** KA plain and Reactant bcast agree; Reactant KA yields incorrect, asymmetric results.

## Performance

```julia
res = 16 * 1024
runme(; nx=res, ny=res, nt=10, use_cuda=true)
```

CUDA GPU (H100), `nx = ny = 16384`, `nt = 10`:

| Backend        | Teff (GB/s) |
|----------------|------------:|
| KA plain       |     3190.74 |
| Reactant KA    |     1669.99 |
| Reactant bcast |     2948.19 |

KA plain reaches peak measured throughput. Reactant bcast is close (~93 %), while Reactant KA currently sits at ~53 %, likely related to the same compilation issue causing the correctness problem above.

## Sharding (weak scaling)

[`diff2D_flux_shard.jl`](diff2D_flux_shard.jl) uses Reactant's IFRT sharding to distribute the global grid across multiple H100 GPUs (2×2 mesh for 4 GPUs). The local grid size is kept fixed (`nx = ny = 16384` per device) so the global problem grows with device count — a weak-scaling test.

```julia
# 1 GPU — diff2D_flux.jl (no sharding, plain Reactant bcast)
runme(; nx=16384, ny=16384, nt=10, use_cuda=true)

# 4 GPUs — diff2D_flux_shard.jl (2×2 mesh, global grid 32768×32768)
runme(; nx=16384, ny=16384, nt=10)
```

> **Note:** the 1-GPU baseline comes from [`diff2D_flux.jl`](diff2D_flux.jl) — no sharding infrastructure involved. The 4-GPU run uses [`diff2D_flux_shard.jl`](diff2D_flux_shard.jl) with Reactant IFRT sharding.

CUDA GPU (H100), `nt = 10`, Reactant bcast:

| Ndev | Script                  | Local grid      | Global grid     | Teff (GB/s) |
|-----:|-------------------------|-----------------|-----------------|------------:|
|    1 | `diff2D_flux.jl`        | 16384 × 16384   | 16384 × 16384   |     2945.88 |
|    4 | `diff2D_flux_shard.jl`  | 16384 × 16384   | 32768 × 32768   |     2408.40 |

**Parallel efficiency:** E_par = 2408.40 / 2945.88 ≈ 81.7 %

The ~18 % overhead at 4 GPUs reflects inter-device communication at shard boundaries and XLA collective overhead.
