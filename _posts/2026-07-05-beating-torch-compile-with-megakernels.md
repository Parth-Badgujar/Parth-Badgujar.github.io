---
layout: post
title: Beating torch.compile with Megakernels in CuTe DSL
date: 2026-07-05 08:00:00
description: A deep dive into custom GPU kernels and how they can outperform torch.compile.
tags: [gpu, cuda, triton, machine-learning, pytorch]
categories: [tech]
mermaid:
  enabled: true
toc:
  sidebar: left
---
<style>
  h2 {
    font-size: 1.5rem !important;
  }
  h3 {
    font-size: 1.25rem !important;
  }
  h4 {
    font-size: 1.1rem !important;
  }

  .highlight pre,
  .highlight code,
  pre code {
    font-size: 0.8rem !important;
    line-height: 1.45 !important;
  }
</style>

*Prior NVIDIA GPU related knowledge is needed before going through this blog. If you're new to the topic, [Modal GPU Glossary](https://modal.com/gpu-glossary) is a great place to start!*

## Intro to Megakernels and Why Megakernels ?

Normally when you run any PyTorch model without any optimizations it runs in `eager` mode, which means each operation is dispatched one by one to the GPU. Adding optimization like `torch.compile` performs operator fusion to reduce the number of kernel launches and improves data-reuse in operations but still you have multiple kernel launches in single forward pass. In megakernels the goal is fuse all operations into a `single` kernel launch.

In the current GPU execution model, on actual hardware you have a limited set of `SMs (Streaming Multiprocessors)` on which blocks of kernels are being scheduled based on the hardware resources used by each block. We'll first device some strategies to design the megakernel.

### Wave Packing

When a kernel has higher number of blocks than SMs can fit, the scheduler launches waves of blocks across all SMs. Say you have kernel with `200 blocks` but the GPU only has `148 SMs`, assuming occupancy of `1 block / SM` it will launch `ceil(200/148) = 2 waves`, so the last wave will only execute `200 % 148 = 52 blocks`, so the remaining `96 SMs` are essentially idle. This is know as `wave quantization`.

<div class="row mt-3 mb-4">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <img src="/assets/img/megakernels/fused_AB_light.excalidraw.svg" class="img-fluid only-light" alt="Standard Dependency vs TMA Overlap">
      <img src="/assets/img/megakernels/fused_AB_dark.excalidraw.svg" class="img-fluid only-dark" alt="Standard Dependency vs TMA Overlap">
    </figure>
  </div>
</div>

Now assume you have two such kernels launched in a sequential manner on the same `cuda stream`, both of them will have extra waves. Through some black magic, if we are able to combine the execution of both kernels together, we can save a complete wave in theory and gain some free lunch.

### Load/Store/Compute Overlap using TMA 

After scheduling the waves properly, to squeeze out maximum performance, we can use `TMA (Tensor Memory Accelerator)` to `async load` the data required in the next wave while we are performing the compute of the current wave. Similarly, we can use TMA to perform `async stores` so that the next wave can run while the current store operation completes. This essentially hides the complete latency of load/store behind compute similar to SoTA `matmul` kernels.

<div class="row mt-3 mb-4">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <img src="/assets/img/megakernels/load_overlap_light.excalidraw.svg" class="img-fluid only-light" alt="Standard Dependency vs TMA Overlap">
      <img src="/assets/img/megakernels/load_overlap_dark.excalidraw.svg" class="img-fluid only-dark" alt="Standard Dependency vs TMA Overlap">
    </figure>
  </div>
</div>

### TMA Compute Overlap (Finegrained) 

The above cases assumed you don't have data dependency between kernels, but otherwise you cannot directly schedule them in parallel, `Kernel B` will require the output of `Kernel A` to be ready before it starts loading. But as we are dealing with machine learning models which have `weights + activations`, we can still prefetch the weights while the activations are not ready and overlap the load with compute.

<div class="row mt-3 mb-4">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <img src="/assets/img/megakernels/finegrained_overlap_light.excalidraw.svg" class="img-fluid only-light" alt="Standard Dependency vs TMA Overlap">
      <img src="/assets/img/megakernels/finegrained_overlap_dark.excalidraw.svg" class="img-fluid only-dark" alt="Standard Dependency vs TMA Overlap">
    </figure>
  </div>
</div>

### Launch Overhead
A kernel launch is not simple a GPU has to setup context for current kernel, clear context of previous kernel, flush the L1/L2 caches and a lot more stuff. Although this takes a couple of microseconds but while doing 100s of passes we can save a couple of milliseconds of total execution time.

By combining all of these strategies into a single `megakernel`, we bypass the standard GPU scheduler and minimize overhead. However, this means we must carefully build our own custom scheduler directly into the kernel, manually managing data dependencies and synchronization across asynchronous execution blocks.


## Implementation Plan

### GPU Architecture
I have access to an `RTX 5070 Ti` (`sm120`), so I decided to optimize for the `sm120` family of GPUs (RTX Pro 6000, RTX 50 Series, and DGX Spark). `sm120`, the so-called "consumer Blackwell," is an interesting architecture in the sense that it borrows hardware features from the `Hopper` family like `TMA`, and also has hardware support for block-scaled matrix multiplication (`NVFP4`, `MXFP4`, etc.), but uses warp-synchronous tensor cores unlike actual Blackwell (`sm100` family) which has async tensor cores and `TMEM (Tensor Memory)`. Apart from the above, there are a lot more differences we'll encounter.

### Model Architecture

* Currently focusing on a simple LLaMA-like (RMSNorm + SwiGLU) transformer architecture. As of now, I haven't included RoPE and the final projection layer from the embedding space to probabilities, currently focusing on the core components of the transformer, will add the rest in future parts. 
* This kernel isn't for direct decode style inference as we have to perform `split-K GEMV` and `split-K attention (flash decoding)` for efficient KV-Cache based decoding. Here I am doing a simple transformer forward pass with KV calculation (without any past KV cache) in compute-bound regime, to showcase the performance benefits. Though the techniques can still be applied to single batch decode kernels.

### Why CuTe DSL?

Majority of DSLs operate on `tile` based abstraction where we are only dealing with tile or vector like data. To design a complex kernel with finegrained hardware management `CuTe DSL` is the way to go, which gives Python flexiblity + CUDA C++'s low level control and directly compiles to `PTX`. I will cover a lot about `CuTe DSL` and kernel profiling in next parts.

## Megakernel Implementation  

### Cooperative Kernel

We launch the megakernel with `gridSize == num SMs`, allocating one block per SM. Each block decodes instructions from global memory, where each instruction represents a unit of `work` containing an operator name (e.g. rmsnorm, matmul, attention) along with arguments like `blockIdx.x`, `blockIdx.y`, and `blockIdx.z` as if the operators were launched as separate kernels. We pre-schedule all instructions in order and assign operator blocks to actual kernel blocks. At runtime, each block reads its assigned instruction and calls the corresponding operator accordingly.

```python
@cute.kernel
def kernel(max_works, mSchedule):
    block_idx = cute.arch.block_idx()[0]
    for work_idx in range(max_works):
        layer_idx    = mSchedule[block_id, work_idx, 0]
        op_kind      = mSchedule[block_id, work_idx, 1]
        pid_m        = mSchedule[block_id, work_idx, 2]
        pid_n        = mSchedule[block_id, work_idx, 3]
        pid_o        = mSchedule[block_id, work_idx, 4]
        expected_cnt = mSchedule[block_id, work_idx, 5]
        current_idx  = mSchedule[block_id, work_idx, 6]
        next_idx     = mSchedule[block_id, work_idx, 7]
  
        if op_kind == int(Op.RMS):
            ...
        elif op_kind == int(Op.QKV):
            ... 
        elif op_kind == int(Op.ATTN):
            ...
        elif op_kind == int(Op.OUT):
            ...
        elif op_kind == int(Op.UP):
            ...
        elif op_kind == int(Op.GATE):
            ...
        elif op_kind == int(Op.DOWN):
            ...
```
### Dependecy Management and Scheduling

The scheduling is done in a simple round robin way across the SMs. 
For eg. we have 8 RMS Norm blocks --> 12 Matmul block --> 8 Attention blocks to be scheduled on 5 SMs we would create the `mSchedule` so that it follows the below diagram. 

<div class="row mt-3 mb-4">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <img src="/assets/img/megakernels/rr_blocks_light.excalidraw.svg" class="img-fluid only-light" alt="Standard Dependency vs TMA Overlap">
      <img src="/assets/img/megakernels/rr_blocks_dark.excalidraw.svg" class="img-fluid only-dark" alt="Standard Dependency vs TMA Overlap">
    </figure>
  </div>
</div>

This eliminates `wave quantization` bubbles and the double warpgroup ping-pong eliminates the compute bubbles and hides maximum load/store latency. But we cannot directly schedule the blocks we have to make sure that the previous output is ready. 

We can manage dependencies using an `atomic counter`. For each block, we first identify all its parent dependencies. When a parent block finishes its computation, it increments the atomic counter at its `next_idx` by 1. The current block polls the atomic counter at its `current_idx`, and as soon as it reaches the predetermined value `expected_cnt`, it begins execution. At the end, each block increments the counter at its own `next_idx` to unblock downstream blocks.

<div class="row mt-3 mb-4 justify-content-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    <figure>
      <img src="/assets/img/megakernels/atomic_demo_light.excalidraw.svg" class="img-fluid only-light" alt="Standard Dependency vs TMA Overlap">
      <img src="/assets/img/megakernels/atomic_demo_dark.excalidraw.svg" class="img-fluid only-dark" alt="Standard Dependency vs TMA Overlap">
    </figure>
  </div>
</div>

### Warpgroup Scheduling and Pipelineing
Each SM can run a maximum of 32 warps (1024 threads), but it can only schedule `4 warps (single warpgroup)` at a time to the actual hardware. Taking that into consideration, we launch 2 warpgroups (= 8 warps) so that while one warpgroup (`wg-1`) is doing its compute, the other warpgroup (`wg-2`) can asynchronously load the data and wait for `wg-1`. As soon as `wg-1` finishes, `wg-2` starts its compute while `wg-1` begins loading data for the next operation similar to ping-pong `matmul` kernels. 


```python
@cute.kernel
def kernel(max_works, mSchedule):
    block_idx = cute.arch.block_idx()[0]
    warp_id    = cute.arch.warp_idx()
    group_id   = warp_id // 4

    for local_work_idx in range(max_works // 2):
        work_idx = local_work_idx * 2 + group_id
        ...   
```

The async load/store handoff logic is written inside each operator. The entire architecture uses **Ampere-style two-stage pipelining with TMA**, two warpgroups operate in ping-pong fashion where **only one** warpgroup is actively computing while the other asynchronously loads data for the next operation. No warp specialization is used, both warpgroups are identical and simply alternate stages with proper barrier handoff and async stores.

I initially considered warp specialization, but chose against it we are already at the brink of register overflow, and dedicating warps to producer/consumer roles would likely push us over. But at the end the kernel used only 233 registers per thread.

### Shared Memory Layout

The `sm120` architecture provides `99 KiB` of max usable shared memory per SM. To ensure smooth, non-blocking handoff between operators, the shared memory is statically partitioned into three regions:

| Region | Size | Purpose |
|--------|------|---------|
| `stage0` buffer | 32 KiB | Active input tile for the computing warpgroup |
| `stage1` buffer | 32 KiB | Prefetch buffer for the next warpgroup's input |
| `output` buffer | 34 KiB | Stores the result of the current operation |
| `mbarriers` + misc | 1 KiB | Barrier objects and miscellaneous metadata |

```python
@cute.jit
def _get_shared_storage(self):
    num_out_elements = max(
        self.bM * (self.bN + self.output_pad),
        self.bQ * (self.head_dim + self.output_pad),
        self.num_sets * self.embed_dim * self.num_stages,
    )

    @cute.struct
    class BarrierStorage:
        load_barrier:    cute.struct.MemRange[Uint64, self.num_stages]
        input_barrier:   cute.struct.MemRange[Uint64, 2]
        output_barrier:  cute.struct.MemRange[Uint64, 2]
        compute_barrier: cute.struct.MemRange[Uint64, 2]
        stage:           cute.struct.MemRange[Int32, 1]
        phase:           cute.struct.MemRange[Int32, 1]

    @cute.struct
    class SharedStorage:
        barriers: BarrierStorage
        stages: cute.struct.Align[cute.struct.MemRange[BFloat16, self.num_stages * self.stage_elements], 128]
        out:    cute.struct.Align[cute.struct.MemRange[BFloat16, num_out_elements], 128]

    return SharedStorage

def kernel(...):
    storage = self._get_shared_storage()
```

While one warpgroup computes using `stage0`, the other prefetches into `stage1`. On the next iteration they swap, this is the classic ping-pong pattern extended uniformly across all operators. Each operator's input tiles must fit within the 32 KiB upper limit of a single stage buffer.

Note that `sm120` does **not** support TMA Swizzled Stores unlike the `sm100` architecture, so we add **16 bytes of padding per row** to the output tiles stored in shared memory to prevent bank conflicts during the store phase.

* **Matmul:** Each stage holds both the A and B tiles in `fp16`. With $\mathrm{blockM} = 64$, $\mathrm{blockN} = 128$, and $\mathrm{blockK} = 64$:

  $$\mathrm{A\ tile} = \mathrm{blockM} \times \mathrm{blockK} \times 2\,\mathrm{B} = 64 \times 64 \times 2 = 8 \;\mathrm{KiB}$$

  $$\mathrm{B\ tile} = \mathrm{blockK} \times \mathrm{blockN} \times 2\,\mathrm{B} = 64 \times 128 \times 2 = 16 \;\mathrm{KiB}$$

  $$\mathrm{Stage\ Size} = 8 + 16 = 24 \;\mathrm{KiB} \leq 32 \;\mathrm{KiB}$$

  The matrix multiply accumulates in `float32` registers but the result is cast back to `fp16` before being written to shared memory, so the output footprint uses 2 bytes per element rather than 4. With 16-byte row padding:

  $$\mathrm{Output\ tile} = \mathrm{blockM} \times (\mathrm{blockN} \times 2\,\mathrm{B} + 16\,\mathrm{B}) = 64 \times 272 = 17408 \;\mathrm{B} = 17 \;\mathrm{KiB} \leq 34 \;\mathrm{KiB}$$

  <div class="row mt-3 mb-4">
    <div class="col-sm mt-3 mt-md-0">
      <figure>
        <img src="/assets/img/megakernels/mm_handoff_light.excalidraw.svg" class="img-fluid only-light" alt="Matmul warpgroup handoff">
        <img src="/assets/img/megakernels/mm_handoff_dark.excalidraw.svg" class="img-fluid only-dark" alt="Matmul warpgroup handoff">
      </figure>
    </div>
  </div>


* **Attention:** I have used Flash Attention v2 like approach, where each stage holds the Q, K, and V tiles. With $\mathrm{blockQ} = 64$, $d_{\mathrm{head}} = 128$, and $\mathrm{blockKV} = 64$ in `fp16`, each tile occupies:

  $$\mathrm{Q\ tile} = \mathrm{blockQ} \times d_{\mathrm{head}} \times 2\,\mathrm{B} = 64 \times 128 \times 2 = 16 \;\mathrm{KiB}$$

  $$\mathrm{K\ tile} = \mathrm{blockKV} \times d_{\mathrm{head}} \times 2\,\mathrm{B} = 64 \times 128 \times 2 = 16 \;\mathrm{KiB}$$

  $$\mathrm{V\ tile} = \mathrm{blockKV} \times d_{\mathrm{head}} \times 2\,\mathrm{B} = 64 \times 128 \times 2 = 16 \;\mathrm{KiB}$$

  However, Q and V are `aliased` in shared memory, Q is loaded once at the start of the attention loop using `cp.async` / `LDGSTS` instruction and is completely loaded into registers after that, so its buffer is reused for V during the K/V streaming phase. At any point during the loop, a single stage holds:

  $$\mathrm{Stage\ Size} = \underbrace{16 \;\mathrm{KiB}}_{\mathrm{Q/V\ (aliased)}} + \underbrace{16 \;\mathrm{KiB}}_{\mathrm{K}} = 32 \;\mathrm{KiB} \leq 32 \;\mathrm{KiB}$$

  The attention output with row padding:

  $$\mathrm{O\ tile} = \mathrm{blockQ} \times (d_{\mathrm{head}} \times 2\,\mathrm{B} + 16\,\mathrm{B}) = 64 \times 272 = 17{,}408 \;\mathrm{B} = 17 \;\mathrm{KiB} \leq 34 \;\mathrm{KiB}$$

  The attention kernel itself is not multi-stage, the KV loop operates within a single stage buffer, but overlaps memory and compute by loading V while the `Q @ K^T matmul` executes, and loading the next K tile while the `P @ V matmul` executes. The two-stage ping-pong only applies across operators: once attention finishes, the next operation begins on the other stage.

  <div class="row mt-3 mb-4">
    <div class="col-sm mt-3 mt-md-0">
      <figure>
        <img src="/assets/img/megakernels/ma_handoff_light.excalidraw.svg" class="img-fluid only-light" alt="Matmul warpgroup handoff">
        <img src="/assets/img/megakernels/ma_handoff_dark.excalidraw.svg" class="img-fluid only-dark" alt="Matmul warpgroup handoff">
      </figure>
    </div>
  </div>

* **RMSNorm:** The row-parallel work distribution is designed to maximize strong scaling across SMs. For N rows, each block is assigned `prev_power_of_two(N / num_sms)` rows to ensure even work distribution. Within each block, a `warps_per_row` parameter controls how many of the 4 available warps cooperate on normalizing a single row, for instance, `warps_per_row = 2` means two rows are in compute simultaneously, each processed by 2 warps. These rows are again two-stage pipelined, one set of rows is being normalized while the next set is being loaded asynchronously 
    <div class="row mt-3 mb-4">
      <div class="col-sm mt-3 mt-md-0">
        <figure>
          <img src="/assets/img/megakernels/rr_handoff_light.excalidraw.svg" class="img-fluid only-light" alt="Matmul warpgroup handoff">
          <img src="/assets/img/megakernels/rr_handoff_dark.excalidraw.svg" class="img-fluid only-dark" alt="Matmul warpgroup handoff">
        </figure>
      </div>
    </div>

Although I have adopted for static shared memory layout you can refer to megakernel by Hazy Research where they have implemented a page based shared memory allocator where each page is 16 KiB and allocated during runtime. But in my case I thought static might be simple with two stages instead of adding extra allocator. Initially I had tried with three stages but that limits the size of `matmul` stages and reduced performance in compute bound regions so I reverted to two stages.

### Synchronization 

To manage such pipeline thereare mainly three barriers namely `input_barrier`, `compute_barrier` and `output_barrier` per warpgroup and an additional set of `load_barrier` one for each stage, in total we have 2x3 + 1x2 = 8 barriers, all of them are `mbarrier` where threads can arrive, wait for other threads or wait for memory transactions.

As there are barriers for each warpgroup I have named them `input_bar_me` (current warpgroup) and `input_bar_ot` (other warpgroup), same scheme for `output_barrier` and `compute_barrier`. 

* **input_barrier:** Sits at the very start of the operator. We wait on the barrier (`input_bar_me`) until the other warpgroup arrives on its `input_bar_ot`. It signals the warpgroup that the input stage is released and now its ready to start loading data. After the barrier there is a `load_stage` variable in shared memory which is the stage to be used in the current iteration. It is updated by the other warpgroup before arrival on `input_bar_ot`.

* **compute_barrier:** `input_barrier` only guarantees that one of the stage buffers is released but not both. Therefore another barrier is required to signal that all stages are now released and we can start computing on the released stage. This barrier is placed just before loading the next pipeline stage.

* **output_barrier:** `output_barrier` guarentees that the output buffer is released so the `wait(output_bar_me)` is placed jut before writing anything to the output buffer and `arrive(output_bar_ot)` is placed after the output is fully stored from SMEM to GMEM.

#### Atomic Spin Lock

Just after the `input_barrier` a single thread spins on the `current_idx` of the atomic array untill its value reaches `expected_cnt`. 

```python
@dataclass
class PipelineMeta:
    current_idx: int
    next_idx: int
    expected_cnt: int

if group_tid == 0: #group_tid = local thread index of the warpgroup
    ready = 0
    while ready != pipeline.expected_cnt:
        ready = ld_acquire_u32((mAtomics.iterator + pipeline.current_idx).toint())
warpgroup_sync()
```

These are the main concepts used in the kernel, after this the remaining part is working with CuTe DSL to actually implement the code, profiling kernels and benchmarking. I did not directly arrive at this architecture it took multiple iterations, errors, race-conditions which were needed to be fixed, I'll explain those nuances in the code section.

## Implementation in CuTe DSL

In CuTe DSL you have to wrap the python functions in `@cute.jit` and `@cute.kernel`. `@cute.jit` has the code which is going to get JIT compiled, it can be either of CPU/GPU function. `@cute.kernel` is the actual entry point of the kernel which is launched with `.launch(grid=(num_sms,), block=(256,))` method of the wrapper. 

### Load / Store Ops

#### LDGSTS and direct GMEM to RMEM loads


#### TMA Copies

I have used TMA for all tensor load / store operations except for the weights of RMSNorm and Attention. Given `N` transformer layers the weights of each operator are stacked contigously, so a weight matrix of shape `(A, B)` is now shaped `(N, A, B)`. This allows use to use a single TMA descriptor for the weights of multiple layers, by adding an extra dim in TMA. 

Then create shared memory layouts of the tensors. The stride of the stage dimention is kept as **32 KiB** and a padding of 8 elements added to `sC`. For padded stores using `TMA` we simply create a shared memory layout which has a larger shape than the global `gC_tile`, then TMA hardware automatically clips the output tile and doesn't write those extra bytes. The normal `sC_layout` does not work with TMA, I guess it needed contiguous shapes and strides. 

```python
sA_layout = cute.make_composed_layout(
    cute.make_swizzle(int(math.log2(bK)) - 3, 4, 3), 0,
    cute.make_ordered_layout(
        shape = (bM, bK),
        order = (1, 0)
    ),
)

sB_layout = cute.make_composed_layout(
    cute.make_swizzle(int(math.log2(bK)) - 3, 4, 3), 0,
    cute.make_ordered_layout(
        shape = (1, bN, bK),
        order = (2, 1, 0)
    ),
)

sC_layout = cute.make_layout(
    shape = (bM, bN),
    stride = (bN + pad, 1) #pad = 8 elements = 16 bytes
)

#Same as the above sC_layout but with contiguous strides
sC_tma_layout = cute.make_ordered_layout(
    shape = (bM, bN + output_pad), order = (1, 0),
)
```

All the TMA atoms are created before the kernel inside the `__call__` function for the input, weights and output activations involved in eacch of the operation. The atoms already have the TMA descriptors embedded in them we don't have to separately create TMA descriptors. For operations where the activations are pointwise added in the epilogue we can use the inplace reductio feature of TMA. 

```python
load_op  = cpasync.CopyBulkTensorTileG2SOp()
store_op = cpasync.CopyBulkTensorTileS2GOp()
if cutlass.const_expr(self.use_tma_reduce):
    store_op_red = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD)
else:
    store_op_red = store_op

# QKV (WS1 @ QKV_w -> WS2)
tma_QKV_inp, g_QKV_inp = cpasync.make_tiled_tma_atom(load_op,  mWS1_embed, sA_layout,  (bM, bK))
tma_QKV_wt,  g_QKV_wt  = cpasync.make_tiled_tma_atom(load_op,  mQKV_proj,  sB_layout,  (1, bN, bK))
tma_QKV_act, g_QKV_act = cpasync.make_tiled_tma_atom(store_op, mQKV_act,   sC_tma_layout, (bM, 1, bN + output_pad))
... #similarly for all operation
```
Note: We cannot pass the raw `mWS1_embed` tensor in the runtime code of the TMA, we have to pass the tensor returned in the above code because it is a special `ArithmeticTuple` tensor which has vectorized strides so that we are able to slice into the tile of the global tensor. During runtime inside the `matmul()` function they below code snippet creates the per thread, partitions of the TMA operation. 

```python
sC_tma = storage.out.get_tensor(sC_tma_layout)

sA_g = cute.group_modes(sA, 0, 2) # (bM, bK, num_stages) -> ((bM, bK), num_stages)
sB_g = cute.group_modes(sB, 0, 2) # (bN, bK, num_stages) -> ((bN, bK), num_stages)
gA_g = cute.group_modes(gA_tile, 0, 2) # (bM, bK) -> ((bM, bK), )
gB_g = cute.group_modes(gB_tile, 0, 3) # (1, bN, bK) -> ((1, bN, bK), )

tAsA, tAgA = cpasync.tma_partition(tma_A, 0, cute.make_layout(1), sA_g, gA_g) # per thread view of sA and gA
tBsB, tBgB = cpasync.tma_partition(tma_B, 0, cute.make_layout(1), sB_g, gB_g) # per thread view of sB and gB
```

We have to group the tile shape mode together for the TMA to interpret the exact tile and rest modes can be used as stages for the async copies. TMA copies are called using `cp.async.bulk.tensor.2d. ...` instruction which is to be called by a single thread asynchronously and pass the tile coordinates, descriptor pointer and the destination address to the instruction. The same thing is done by `cute.copy(...)` function below. `cute.copy(...)` with TMA atom internally calls `cute.elect_one()` which elects a single thread from a warp to run that instruction therefore we have to wrap it in `if warp_id == 0:`, if you pass `cute.copy(...)` in `elect_one()` then it will stall the execution because of calling nested `elect_one()`, this is a common pitfall to avoid. 
```python
if warp_id == 0:
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(load_barrier + stage_idx, 2) # increment transcation count
    cute.copy(tma_A, tAgA[None, tile_idx], tAsA[None, stage_idx], tma_bar_ptr = load_bar + stage_idx)
```

For TMA stores we have to perform the same operations `groupmodes -> partition -> copy`, but TMA stores don't support mbarrier completition mechanism, therefore we also have to call the `bulk_group` API to commit the store and track it. Additionally I have called `fence_proxy_async_global()` to ensure that the stores are visible to other SMs, without this you will constantly have race conditions within SMs, it took me and claude a while to figure this out.

```python
if warpgroup.warp_id == 0:
    gC_tma_tile = cute.local_tile(gC_tma, (bM, 1, bN + output_pad), (pid_m, pid_n, 0))
    sC_g     = cute.group_modes(sC_tma,      0, cute.rank(sC_tma.layout))
    gC_tma_g = cute.group_modes(gC_tma_tile, 0, cute.rank(gC_tma_tile.layout))
    sC_part, gC_part = cpasync.tma_partition(tma_C, 0, cute.make_layout(1), sC_g, gC_tma_g)
    cute.copy(tma_C, sC_part, gC_part)
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0)
    fence_proxy_async_global()
```

#### LdMatrix / StMatrix Instructions 

These copies are similar to most of the copy atoms but you have to be carefull when to use the transpose and when not to use transpose and the datatype. CuTe DSl actually simplifies the usage of this API by handling the address generation and applying the swizzling automatically, or else it was a huge pain to use the `ldmatrix` and `stmatrix` instructions with inline ptx. 

```python
# group_tidx is thread index within a warp group
tiled_mma = cute.make_tiled_mma(
    warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
    (warpM, warpN, 1),
    permutation_mnk = (bM, bN, bK),
)

thr_mma = tiled_mma.get_slice(warpgroup.group_tidx)

# this seemse confusing but it creates registers tensors based on per-thread partitioned shared memory
tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0])) 
tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))

ldmatrix = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
thr_copy_A = cute.make_tiled_copy_A(ldmatrix, tiled_mma).get_slice(warpgroup.group_tidx)
thr_copy_B = cute.make_tiled_copy_B(ldmatrix, tiled_mma).get_slice(warpgroup.group_tidx)

tCsA = thr_copy_A.partition_S(sA[None, None, stage_idx])
tCsB = thr_copy_B.partition_S(sB[None, None, stage_idx])

# reshape according to ldmatrix (only for copy)
tCrA_cpy = thr_copy_A.retile(tCrA)
tCrB_cpy = thr_copy_B.retile(tCrB)

cute.copy(thr_copy_A, tCsA, tCrA_cpy)
cute.copy(thr_copy_B, tCsB, tCrB_cpy)

cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```
<div class="row mt-3 mb-4">
    <div class="col-sm mt-3 mt-md-0">
      <figure>
        <img src="/assets/img/megakernels/gemm_ldmatrix_light.excalidraw.svg" class="img-fluid only-light" alt="GEMM_Ldmatrix">
        <img src="/assets/img/megakernels/gemm_ldmatrix_dark.excalidraw.svg" class="img-fluid only-dark" alt="GEMM_Ldmatrix">
      </figure>
    </div>
  </div>


Similarly for storing we have to use the `stmatrix` atom and `cute.make_tiled_copy_C` and rest is similar.

```python
thr_mma = tiled_mma.get_slice(warpgroup.group_tidx)
store_atom = cute.make_copy_atom(
    cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16
)
thr_copy_C = cute.make_tiled_copy_C(stmatrix, tiled_mma).get_slice(warpgroup.group_tidx)
```
