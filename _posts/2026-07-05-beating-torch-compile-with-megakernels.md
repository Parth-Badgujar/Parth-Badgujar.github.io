---
layout: post
title: Beating torch.compile with Megakernels in CuTe DSL [Part 1]
date: 2026-07-05 08:00:00
description: A deep dive into custom GPU kernels and how they can outperform torch.compile.
tags: [gpu, cuda, triton, machine-learning, pytorch]
categories: [tech]
mermaid:
  enabled: true
toc:
  sidebar: left
---

## Intro to Megakernels and Why Megakernels ?

If we look at the current GPU execution model and architecture, on actual hardware you have limited set of `SMs (Streaming Multiprocessors)` on which blocks of kernels are being scheduled based on the resources used by the kernel. Lets look at some of the strategies we can use to design our megakernel.

* **Wave Packing:** By default, kernels on the same `cuda stream` run strictly sequentially, forcing the hardware scheduler to execute them in isolated, distinct waves. But if, through some black magic, we can fuse both kernels into a single `megakernel`, the scheduler can pack the grid much more efficiently. As shown below, eliminating the strict boundary between the two kernels allows the GPU to overlap their tails and heads—saving an entire execution wave!

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <style>
        /* Hide mermaid gantt chart vertical grid lines */
        .mermaid .grid, .mermaid .grid-line, .mermaid .tick line {
          display: none !important;
        }
        /* Highlight the second section (optimized) background */
        .mermaid rect.section.section1 {
          fill: rgba(6, 137, 173, 0.45) !important; /* Light green */
        }
        /* Reduce gap below mermaid diagrams */
        .mermaid {
          margin-bottom: -1rem !important;
        }
        figure {
          margin-bottom: 0 !important;
        }
        /* Reduce heading sizes to match the original h3/h4 sizes */
        h2 {
          font-size: 1.75rem !important;
        }
        h3 {
          font-size: 1.5rem !important;
        }
      </style>
      <div class="mermaid">
      gantt
          title Standard Sequential vs Megakernel Execution (144 SMs)
          dateFormat  X
          axisFormat %s

          section Sequential (4 Waves)
          Kernel A (144 blocks) : 0, 1
          Kernel A (56 blocks)  : 1, 2
          Kernel B (144 blocks) : 2, 3
          Kernel B (56 blocks)  : 3, 4

          section Megakernel (3 Waves)
          Kernel A (144 blocks)         : 0, 1
          Kernel A (56) + Kernel B (88) : 1, 2
          Kernel B (112 blocks)         : 2, 3
      </div>
    </figure>
  </div>
</div>

* **TMA Compute Overlap:** The above case assumed you don't have any data dependency between `Kernel A` and `Kernel B`, but if you have any dependency you cannot schedule both of them together you'll have to wait for first kernel to finish. But still if we wish to squeeze the maximum performance we can use `TMA (Tensor Memory Accelerator)` to overlap the execution of first kernel with the data loading of second kernel.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <div class="mermaid">
      gantt
          title Standard Dependency vs TMA Overlap
          dateFormat  X
          axisFormat %s

          section Standard (Sequential)
          Kernel A (Compute)      : a1, 0, 2
          Kernel B (Memory Load)  : a2, 2, 3
          Kernel B (Compute)      : a3, 3, 5

          section TMA Overlap
          Kernel A (Compute)      : b1, 0, 2
          TMA Load (Kernel B Data): b2, 1, 2
          Kernel B (Compute)      : b3, 2, 4
      </div>
    </figure>
  </div>
</div>

* **TMA Compute Overlap (Finegrained):** But you might ask, How will I load the data if it is not yet ready by `Kernel A` ? Well we are dealing with machine learning models each layer has weights + activations, a kernel needs to load both of them before computation, even if we are not able to load the activations we can still prefetch the weights of the next kernel.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <figure>
      <div class="mermaid">
      gantt
          title Finegrained TMA Overlap (Prefetching Weights)
          dateFormat  X
          axisFormat %s

          section Standard
          Kernel A (Compute)          : a1, 0, 3
          Kernel B (Load Weights)     : a2, 3, 4
          Kernel B (Load Activations) : a3, 4, 5
          Kernel B (Compute)          : a4, 5, 7

          section TMA Finegrained
          Kernel A (Compute)          : b1, 0, 3
          TMA Load (Kernel B Weights) : b2, 1, 3
          Kernel B (Load Activations) : b3, 3, 4
          Kernel B (Compute)          : b4, 4, 6
      </div>
    </figure>
  </div>
</div>

* **Launch Overhead:** A kernel launch is not simple a GPU has to setup context for current kernel, clear context of previous kernel, flush the L1/L2 caches and a lot more stuff. Although this takes a couple of microseconds but while doing 1000s of passes we can save a couple of milliseconds of total execution time.

By combining all of these strategies into a single `megakernel`, we bypass the standard GPU scheduler and minimize overhead. However, this means we must carefully build our own custom scheduler directly into the kernel, manually managing data dependencies and synchronization across asynchronous execution blocks.

## Why CuTe DSL ?

ML systems is going through very intresting times, earlier everything use to be in CUDA C++ but now you have 10 different ways of writing GPU kernels thanks to `DSLs (Domain Specefic Languages)`. Majority of DSLs operate on `tile` based abstraction where we are only dealing with tile or vector like data. To design a complex kernel with finegrained hardware management `CuTe DSL` is the way to go, which gives Python flexiblity + CUDA C++'s low level control and directly compiles to `PTX`. I will cover a lot about `CuTe DSL` so you can use this as a CuTe DSL guide.


## Implementation Plan

### GPU Architecture
I have access to `RTX 5070 Ti` (`sm120`) therefore I decided to optimize for the `sm120` family of GPUs (RTX Pro 6000, RTX 50 Series and DGX Spark). `sm120` the so called `consumer blackwell` is an instresting architecture in a sense that it has hardware features from `hopper` family like `TMA`, also hardware support for block-scaled matrix multiplication (`NVFP4`, `MXFP4`, etc.) but has warp synchronous tensor cores unlike actual blackwell (`sm100` family) which has async tensor cores and `TMEM (Tensor Memory)`. There are a lot more differences which we will encounter in the blog series.

### Model Architecture

Currently focusing on a simple LLaMA-like transformer architecture. As of now, I haven't included RoPE and the final projection layer from the embedding space to probabilities, currently focusing on the core components of the transformer, will add the rest in future parts.

### Megakernel Implementation

The kernel wil be a cooperative kernel where the grid size if number of SMs so that each block will be given a single SM. On each block we will have 8 active warps out of that 4 warps 