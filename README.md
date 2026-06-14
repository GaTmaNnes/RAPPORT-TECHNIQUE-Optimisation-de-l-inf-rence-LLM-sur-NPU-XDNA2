============================================================================
RAPPORT TECHNIQUE : Optimisation de l'inférence LLM sur NPU XDNA2
============================================================================

1. EXECUTIVE SUMMARY
Standard quantization (INT4) is often theorized as a compute accelerator. 
Our empirical measurements on XDNA2 reveal that throughput is governed 
by geometric constraints, not precision. We report a 35-70% gain using 
"Hardware Fitness" optimization.

2. EMPIRICAL PERFORMANCE ANALYSIS
- Stagnation: INT4 vs INT8 shows no significant TPS gain (Compute Starvation).
- Alignment: 99% correlation between (hidden_size % 512 == 0) and NPU efficiency.
- Bottleneck: Underutilization of NPU tiles occurs on non-aligned models.



3. D2-PRODUCTION METHODOLOGY
Objective: Maximize service value per layer (i):
Max Σ (x_i,q * (TPS_baseline_i * α_i + λ * Mem_Gain_q))
- α_i (Efficiency Index): Measured penalty factor for misalignment.
- λ: Balancing parameter for memory footprint vs. speed.

4. RECOMMENDED HYBRID DEPLOYMENT
- NPU: Prioritize layers with hidden_size as a multiple of 512.
- GPU (RTX 5070): Offload critical layers (lm_head) or non-aligned tensors.
- Validation: Hardware Fitness profiling for every new model compilation.

5. CONCLUSION
Performance on XDNA2 is a matter of data geometry. By optimizing for tile 
alignment rather than theoretical precision, we bypass standard bottlenecks.

