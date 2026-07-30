# Phase-2 Comprehensive Operator Crawl Ledger (2026-07-20 ->)

Goal: systematically populate the KernelWiki knowledge base with H200-validated
SM90 operator implementations, common-first, for the kernel-design-agent. Each
item: crawl a real source (raw.githubusercontent / Gitee raw, fetched via the
H200 remote since WSL HTTP is blocked) OR write a canonical Triton/CUDA impl
citing the originating PR/doc; ALWAYS validate on H200 (correctness + latency);
annotate PURPOSE = speedup | robustness | both + a one-line "what it does".
promote = new/updated wiki page + result log + derived bundle (PROVENANCE sha256)
+ an `operator_purpose` field. record-negative and defer are valid.

## Hard rules (same as phase 1, + purpose annotation)
- H200 = SM90 (Hopper). Only SM90-runnable. SM100-only (tcgen05/TMEM/NVFP4/2-SM) -> DEFER.
- First each firing: `/home/kirin_14379/projects/ai4qz/.venv/bin/ai4qz check h200_ncu`. Not ok -> NOTE + END.
- Crawl path (WSL HTTP is blocked): fetch source via the H200 remote
  (`ai4qz run h200_ncu --cmd 'curl -sSL <raw-url> -o /tmp/x && ...'`) or MCP
  web_reader. Gitee raw anonymous; raw.githubusercontent anonymous (rate-limited);
  GitCode needs PAT (unavailable) -> skip GitCode unless a public raw URL works.
  Never clone whole repos; record URL + SHA256 in the run log.
- Validate: correctness (vs torch / cuBLAS reference) + repeated latency (CUDA
  events, warmup+iters). Compare baseline vs optimized where applicable. Record
  shapes/counts/latency/ratio. NEVER fabricate a speedup.
- Each promote MUST include `operator_purpose: speedup|robustness|both` and a
  `what_it_does:` one-liner in the wiki page + result log.
- Checkpoint EVERY firing: `python3 scripts/validate.py` (0 errors) +
  `python3 scripts/generate-indices.py`. Then mark the item here with a one-line
  result incl. purpose + measured ratio. ~1-2 operators/firing then END.
- NO commits/push/deletes/clone. `rm` blocked -> mv. captured_at for NEW source
  pages '2026-04-27'; H200 dates '2026-07-20 (phase2)'.
- DO NOT STOP when the worklist exhausts: append more operators from the
  category backlog below (or new common operators). Keep going until the USER
  says stop. Only the user can stop this run.

## Category backlog (common -> less common; check off as done; append more as needed)
NORM: rmsnorm-fwd, layernorm-fwd, fused-add-rmsnorm, rmsnorm-bias-residual, rmsnorm-fp32-reduce(robustness)
ACTIVATION: silu-and-mul, gelu-tanh, gelu-exact, swiglu-gated
SOFTMAX: online-softmax-fwd, fused-temp-softmax(robustness NaN/-inf), softmax-bwd
GEMM: bf16-gemm-vs-cublas, tf32-gemm-vs-cublas, fused-gemm-bias-act, grouped-gemm, splitk-gemm, low-mem-gemm(robustness)
REDUCTION: block-warp-sum-reduce, variance-reduce, argmax-row, cooperative-grid-reduce
QUANT: per-tensor-w8-quant, per-channel-quant, per-token-dynamic-quant, fp8-e4m3-quant, int4-dequant-w4a16, groupwise-quant, fp8-oob-clamp(robustness)
ATTENTION: fa2-causal, gqa-mqa-attn, sliding-window-attn, rmsnorm-fp8quant-fused, paged-decode-attn
COPY/LAYOUT: tiled-transpose-square, tiled-transpose-nonsquare, swizzle-layout, concat-split
SCAN/SORT: prefix-sum-scan, topk-selection, radix-sort-small
MOE: moe-topk-gating, moe-permute-unpermute, moe-grouped-gemm-combine
SPARSE/CONV: block-sparse-matmul, depthwise-conv, implicit-gemm-conv, max-pool
ROBUSTNESS: nan-safe-softmax, fp8-oob-clamp-quant, layernorm-fp32-overflow, mixed-prec-gemm-accum


- [DONE] layernorm-fwd (phase2): purpose=both; correctness PASS; Triton fused 1.10-1.51x faster than torch layer_norm. + wiki/kernels/layernorm-hopper.md, op_layernorm.py, op-layernorm-h200-results.md.
## Progress (most recent first)
- [DONE] fused-qkt-scale-mask (phase2, appended): 2.15-5.49x torch (attention score GEMM epilogue).
- [DONE] fused-sigmoid-mul (phase2, appended): 1.37-1.67x torch (GRU/LSTM gate).
- [DONE] fused-addcmul (phase2, appended): 0.86-0.98x torch (~parity).
- [DONE] fused-clamp-scale (phase2, appended): 1.61-1.86x torch.
- [DONE] fused-prenorm-residual (phase2, appended): 2.23-2.86x torch.
- [DONE] fused-add-silu (phase2, appended): 1.36-1.66x torch.
- [DONE] fused-ln-gelu (phase2, appended): 1.81-2.29x torch.
- [DONE] fused-rmsnorm-rope (phase2, appended): 1.77-2.74x torch.
- [DONE] group-norm (phase2, appended): purpose=speedup; PASS err ~1e-6; 1.48-5.29x torch.
- [DONE] sigmoid-fwd (phase2, appended): purpose=speedup(record-negative); PASS; 0.85-0.99x (~parity).
- [COVERED] rmsnorm-fp32-reduce + swiglu-gated (old worklist items covered by existing pages).
- [DONE] kl-div (phase2, appended): purpose=speedup; PASS err ~3e-8; 2.96-3.07x moderate N.
- [DONE] cumprod (phase2, appended): purpose=characterization; PASS rel ~2e-6; 1.50-3.28x.
- [DONE] causal-mask-gen (phase2, appended): purpose=both; correctness PASS; 1.22-5.13x torch (direct write vs ones+mul+triu). wiki/kernels/causal-mask.md.
- [DONE] bce-loss (phase2, appended): purpose=speedup; correctness PASS (err ~5e-7); fused BCE-with-logits 3.03-3.74x faster than torch. wiki/kernels/bce-loss.md.
- [DONE] grad-clip-norm (phase2, appended): purpose=both; correctness PASS (norm err ~3e-7); 0.31-0.40x torch (2-kernel slower). wiki/kernels/grad-clip.md.
- [DONE] adam-step (phase2, appended): purpose=speedup; correctness PASS (fp32 ~3e-8); fused AdamW 2.56-3.05x faster than torch (3R+3W vs ~8 separate ops). wiki/kernels/adam-step.md.
- [DONE] cosine-similarity (phase2, appended): purpose=speedup; correctness PASS (err ~1e-8); 5.26-6.67x torch (fused dot+norms single-pass). wiki/kernels/cosine-sim.md.
- [DONE] dropout (phase2, appended): purpose=speedup(record-negative); correctness PASS (statistical); 0.54-0.85x torch (torch native dropout faster). wiki/kernels/dropout.md.
- [DONE] silu-bwd (phase2, appended): purpose=speedup; correctness PASS (fp32 ~7e-7, bf16 ~4e-3); 1.02-1.29x torch autograd. wiki/kernels/silu-bwd.md.
- [DONE] relu-fwd-bwd (phase2, appended): purpose=speedup(characterization); bit-identical; fwd ~0.9x, bwd 1.0-1.35x. wiki/kernels/relu.md.
- [DONE] l2-normalize (phase2, appended): purpose=speedup; correctness PASS (fp32 ~1e-8, bf16 ~2e-4); 1.95-3.31x torch. wiki/kernels/l2-normalize.md.
- [DONE] gelu-bwd (phase2, appended): purpose=speedup; correctness PASS (fp32 ~1e-6, bf16 ~1e-4); 1.01-1.27x torch autograd. wiki/kernels/gelu-bwd.md.
- [DONE] matmul-bwd (phase2, appended): purpose=speedup(characterization); correctness OK (bf16 noise); ~0.83-1.05x cuBLAS (2 strided GEMMs; backward-only timed). wiki/kernels/matmul-bwd.md.
- [DONE] mixed-prec-gemm-accum (phase2): purpose=robustness; fp32-acc ~1e-5 rel vs bf16-acc 2.6->12.9% (3000-7000x worse, grows with K). Robustness section added to wiki/kernels/bf16-gemm-hopper.md.
- [DONE] scatter-add/index_add (phase2, appended): purpose=speedup; correctness PASS (rel ~6e-8); 1.10-1.44x torch.index_add. wiki/kernels/scatter-add.md.
- [DONE] layernorm-fp32-overflow (phase2): purpose=robustness; fp32 reduction ~0.25% rel vs bf16 0.48->3.64% (8x worse at N=57344). Robustness section added to wiki/kernels/layernorm-hopper.md.
- [DONE] cross-entropy-loss (phase2, appended): purpose=speedup; correctness PASS (err 1.9e-6); 2.03-2.43x moderate vocab, 0.34x D=128K. wiki/kernels/cross-entropy.md.
- [DONE] embedding-lookup (phase2): purpose=speedup(record-negative); bit-identical; ~parity torch (memcpy-bound). wiki/kernels/embedding-lookup.md. nan-safe-softmax covered by fused-temp-softmax.
- [DONE] max-pool (phase2): purpose=speedup; bit-identical; 0.54-0.68x small spatial, 1.24-1.45x large. wiki/kernels/max-pool.md.
- [DONE] fused-gated-mlp (phase2): purpose=speedup(record-negative); correctness OK (bf16 noise); 0.80-0.90x torch 2-GEMM seq. wiki/kernels/fused-gated-mlp.md.
- [DONE] batched-matmul bmm (phase2): purpose=speedup(record-negative); correctness PASS (err=0); 0.49-0.69x cuBLAS (naive Triton vs strided-batched cuBLAS). wiki/kernels/batched-gemm.md.
- [DONE] addmm-fused (phase2): purpose=speedup(characterization); correctness PASS; 0.86-0.95x cuBLAS (torch.addmm already fused). wiki/kernels/addmm-fused.md.
- [DONE] layernorm-bwd (phase2): purpose=speedup(marginal); correctness PASS moderate N (grad_x ~3e-6); 1.06-1.15x moderate, 0.21-0.26x large N (spill+atomics+1-pass var). wiki/kernels/layernorm-bwd.md.
- [DONE] rmsnorm-bwd (phase2): purpose=speedup; correctness PASS (grad_x ~2e-6, grad_w ~1e-3); 1.34-1.66x moderate N, 0.28-0.35x large N (BLOCK spill + atomic). wiki/kernels/rmsnorm-bwd.md.
- [DONE] softmax-bwd (phase2): purpose=speedup; correctness PASS (rel ~7e-8); 3.35-4.04x moderate N, 0.70x N=32000. wiki/kernels/softmax-bwd.md.
- [DONE] rotary-pos-emb RoPE (phase2): purpose=speedup; correctness PASS (err 4.9e-4); 1.66-5.61x torch (fused rotate-half vs cat/slice). wiki/kernels/rope.md.
- [DONE] depthwise-conv (phase2): purpose=speedup; bit-identical; ~parity small spatial, 1.53-1.76x faster large. wiki/kernels/depthwise-conv.md.
- [DONE] block-sparse-matmul (phase2): purpose=speedup(record-negative); correctness PASS (err=0); naive random-tile block-sparse ~0.01x dense (L2 thrash); needs cuSPARSELt/FlashInfer-style tile sorting. wiki/kernels/block-sparse-matmul.md.
- [DONE] moe-permute-unpermute (phase2): purpose=speedup(record-negative); bit-identical; ~parity torch gather/scatter (memcpy-bound). wiki/kernels/moe-permute.md.
- [DONE] moe-topk-gating (phase2): purpose=speedup; correctness PASS (werr 1e-7, idx set_match=1.0); fused top-k+softmax 1.29-3.69x torch. wiki/kernels/moe-gating.md.
- [DONE] topk-selection (phase2): purpose=speedup; correctness PASS (val_err=0, set_match=1.0); 3.54-4.14x moderate N, 0.57-0.58x N=32000. wiki/kernels/topk.md.
- [DONE] prefix-sum-scan (phase2): purpose=speedup; fp32 correctness PASS (rel 1e-6); 2.9-4.0x moderate N, 0.31-0.35x N=32000 (BLOCK spill). wiki/kernels/prefix-scan.md.
- [DONE] concat-split (phase2): purpose=speedup(record-negative); bit-identical; ~parity torch.cat (memcpy-bound), 0.22x at fp32 Dk=14336 large-tile. wiki/kernels/concat-split.md.
- [DONE] tiled-transpose (square+nonsquare) (phase2): purpose=speedup; bit-identical; 2.79-6.01x faster than torch (fp16 gains more). wiki/kernels/tiled-transpose.md. swizzle-layout deferred-to-existing wiki/techniques/swizzling.md.
- [DONE] sliding-window-attn (phase2): purpose=both; correctness PASS (err 2.4e-4); 6.9-14.3x naive O(M^2) ref; finite-neg masking avoids -inf-(-inf) NaN; window-capping deferred. wiki/kernels/sliding-window-attn.md.
- [DONE] gqa-mqa-attn (phase2): purpose=speedup; correctness PASS (err 3.8-7.6e-6); 0.27-0.47x torch SDPA (GQA head-group mapping correct, avoids KV replication; naive kernel loses to optimized backend). wiki/kernels/gqa-mqa-attn.md.
- [DONE] fa2-causal (phase2): purpose=speedup; correctness PASS (err 3-6e-5); 0.36-0.66x torch SDPA (naive Triton FA-2 slower than torch backend). Causal section added to wiki/kernels/triton-fa2-hopper.md.
- [DONE] per-token-dynamic-quant (phase2): purpose=both; correctness PASS; 6.33-12.37x faster than torch (SmoothQuant-style dynamic per-token). wiki/kernels/per-token-dynamic-quant.md.
- [DONE] fp8-oob-clamp (phase2): purpose=robustness; clamped fp8 cast = 0 NaN (max 448) vs naive 64 NaN on outliers/Inf; clamp overhead ~1.6% (~free). Attached robustness section to wiki/kernels/fp8-quant.md.
- [DONE] groupwise-quant (phase2): purpose=both; correctness PASS (scales exact); 2.70-3.42x faster than torch (G=128 AWQ/GPTQ granularity). wiki/kernels/groupwise-quant.md.
- [DONE] int4-dequant-w4a16 (phase2): purpose=speedup(memory); correctness bit-identical; 19-37x faster than torch (4x weight-mem reduction, W4A16). wiki/kernels/int4-dequant.md.
- [DONE] fp8-e4m3-quant (phase2): purpose=speedup; correctness PASS (~99.996% exact); 2.0-3.06x faster than torch at large N (enables FP8 GEMM). wiki/kernels/fp8-quant.md.
- [DONE] per-channel-quant (phase2): purpose=both; correctness PASS (scales exact); fused 6.25-12.08x faster than torch. wiki/kernels/per-channel-quant.md.
- [DONE] per-tensor-w8-quant (phase2): purpose=speedup; correctness PASS (fp32 exact, bf16 maxdiff<=1); 2.3-3.6x faster than torch at large N (enables INT8 GEMM). wiki/kernels/per-tensor-quant.md.
- [DONE] argmax-row (phase2): purpose=speedup; correctness 100% match; 1.10-1.76x torch.argmax. wiki/kernels/argmax.md.
- [DONE/covered] variance-reduce (phase2): inline variance reduction already validated in kernel-layernorm-hopper (sum_x/sum_x2 one-pass). Marked covered, no new page.
- [DONE] block-warp-sum-reduce (phase2): purpose=speedup(record-negative); correctness PASS (rel~1e-7); 0.62-0.95x torch.sum. wiki/kernels/block-reduce.md.
- [DONE] splitk-gemm (phase2): purpose=speedup; correctness PASS; 1.26-2.20x tall-K/small-MN (best 2.20x at 128x128 K=16384), 0.77x at 512x512. wiki/kernels/splitk-gemm.md.
- [DONE] grouped-gemm (phase2): purpose=speedup; correctness PASS (bit-identical); 0.85-1.39x vs torch loop (biggest at many small expert groups). wiki/kernels/grouped-gemm-hopper.md. (fixed 1D-grid N-tiling bug.)
- [DONE] fused-gemm-bias-act (phase2): purpose=speedup; correctness PASS; fused GEMM+bias+silu 1.04-1.17x faster than torch. wiki/kernels/fused-gemm-bias-act.md.
- [DONE] tf32-gemm-vs-cublas (phase2): purpose=speedup(record-negative); correctness PASS; naive Triton TF32 ~8% peak (0.2-0.3x cuBLAS); use cuBLAS. wiki/kernels/tf32-gemm-hopper.md.
- [DONE] bf16-gemm-vs-cublas (phase2): purpose=speedup(baseline); correctness PASS; Triton 674-715 TFLOPS (~72% bf16 peak), 0.89-0.99x cuBLAS at large. wiki/kernels/bf16-gemm-hopper.md.
- [DONE] fused-temp-softmax (phase2): purpose=both; correctness PASS (incl -inf mask); fused 2.0-5.5x faster than torch. wiki/kernels/fused-temp-softmax.md.
- [DONE] online-softmax-fwd (phase2): purpose=both; correctness PASS; 1.2-3.4x faster at N<=32K, 0.23x at N=128K (BLOCK_N=131072 spills; note: tile for huge N). wiki/kernels/online-softmax.md.
- [DONE] gelu-tanh (phase2): purpose=speedup(marginal); correctness PASS; ~parity 0.87-1.12x torch (gelu already fused). wiki/kernels/gelu.md.
- [DONE] silu-and-mul (phase2): purpose=speedup; correctness PASS; fused 1.30-1.69x faster than torch (silu+mul). + wiki/kernels/silu-and-mul.md.
- [DONE] fused-add-rmsnorm (phase2): purpose=speedup; correctness PASS; fused 1.10-1.45x faster than torch (add+rmsnorm+copy). + wiki/kernels/fused-add-rmsnorm.md.
- [DONE] rmsnorm-fwd (phase2): purpose=both; correctness PASS (dtype-precision); Triton fused 1.0-1.40x faster than torch rms_norm. + wiki/kernels/rmsnorm-hopper.md, op_rmsnorm.py, op-rmsnorm-h200-results.md. Added operator tags to data/tags.yaml.

## Worklist (claim next unfinished [ ]; mark [x] when done/deferred)

### NORM
- [x] rmsnorm-fwd:  -> DONE both; correctness PASS; Triton 1.0-1.40x faster than torch rms_norm. wiki/kernels/rmsnorm-hopper.md. Triton RMSNorm forward vs torch.nn.functional.rms_norm; purpose=both (fused speedup + eps/numerical). NEW wiki/kernels/rmsnorm-hopper.md.
- [x] layernorm-fwd:  -> DONE both; correctness PASS; Triton 1.10-1.51x faster than torch layer_norm. wiki/kernels/layernorm-hopper.md. Triton LayerNorm forward vs torch.nn.LayerNorm; purpose=both. wiki/kernels/layernorm-hopper.md.
- [x] fused-add-rmsnorm:  -> DONE speedup; correctness PASS (out bf16 3.1e-2, residual bit-identical); fused 1.10-1.45x faster than torch. wiki/kernels/fused-add-rmsnorm.md. fused (residual+x)+RMSNorm vs two-op; purpose=speedup (fusion). wiki/kernels/fused-add-rmsnorm.md.
- [x] rmsnorm-fp32-reduce:  -> COVERED by layernorm-fp32-overflow (same robustness principle; all norm kernels already use fp32 reduction). RMSNorm with fp32 reduction (avoid fp16 overflow); purpose=robustness. attach to rmsnorm-hopper.md.

### ACTIVATION
- [x] silu-and-mul:  -> DONE speedup; correctness PASS; fused 1.30-1.69x faster than torch. wiki/kernels/silu-and-mul.md. LLM MLP silu(x)*y (gate*up) vs torch; purpose=speedup (fusion). wiki/kernels/silu-and-mul.md.
- [x] gelu-tanh:  -> DONE speedup(marginal); correctness PASS (bf16 bit-identical); ~parity 0.87-1.12x torch. wiki/kernels/gelu.md. (tl.tanh absent in 3.6 -> tanh via exp) GELU tanh-approx vs torch; purpose=speedup. wiki/kernels/gelu.md.
- [x] swiglu-gated:  -> COVERED by fused-gated-mlp (SwiGLU = fused gate-up + silu*mul). SwiGLU gated activation; purpose=speedup. wiki/kernels/swiglu.md.

### SOFTMAX
- [x] online-softmax-fwd:  -> DONE both; correctness PASS; 1.2-3.4x faster at N<=32K but 0.23x at N=128K (BLOCK_N spill; need tiled softmax). wiki/kernels/online-softmax.md. online softmax (one pass, no materialize) vs torch; purpose=both. wiki/kernels/online-softmax.md.
- [x] fused-temp-softmax:  -> DONE both; correctness PASS (incl -inf mask); fused 2.0-5.5x faster than torch. wiki/kernels/fused-temp-softmax.md. fused temperature scale + softmax with -inf/NaN safety; purpose=both (speedup + robustness). attach to online-softmax or sampling page.

### GEMM
- [x] bf16-gemm-vs-cublas:  -> DONE speedup(baseline); correctness PASS; Triton 674-715 TF (~72% peak), 0.89-0.99x cuBLAS. wiki/kernels/bf16-gemm-hopper.md. Triton bf16 GEMM vs cuBLAS, characterize TFLOPS/utilization; purpose=speedup baseline. wiki/kernels/bf16-gemm-hopper.md.
- [x] tf32-gemm-vs-cublas:  -> DONE speedup(record-negative); correctness PASS (TF32 rounding); naive Triton ~82TF (8% peak), 0.2-0.3x cuBLAS. wiki/kernels/tf32-gemm-hopper.md. Triton TF32 GEMM vs cuBLAS; purpose=speedup baseline. attach to bf16-gemm or new.
- [x] fused-gemm-bias-act:  -> DONE speedup; correctness PASS; fused 1.04-1.17x faster than torch (epilogue fusion). wiki/kernels/fused-gemm-bias-act.md. GEMM+bias+SiLU fused vs sequential; purpose=speedup (fusion). wiki/kernels/fused-gemm-bias-act.md.
- [x] grouped-gemm:  -> DONE speedup; correctness PASS (bit-identical); modest 0.85-1.39x vs torch loop (biggest at many small expert groups). wiki/kernels/grouped-gemm-hopper.md. (fixed a 1D-grid bug -> 2D grid) variable-size grouped GEMM (MoE-relevant) vs loop of GEMMs; purpose=speedup. wiki/kernels/grouped-gemm-hopper.md.
- [x] splitk-gemm:  -> DONE speedup; correctness PASS (err=0.0); 1.26-2.20x for tall-K/small-MN, 0.77x at 512x512. wiki/kernels/splitk-gemm.md. split-K GEMM for tall-K vs single; purpose=speedup. wiki/kernels/splitk-gemm.md.

### REDUCTION
- [x] block-warp-sum-reduce:  -> DONE speedup(record-negative); correctness PASS (rel~1e-7); 0.62-0.95x torch.sum (use torch/CUB for standalone reduce). wiki/kernels/block-reduce.md. block+warp sum reduction vs torch.sum; characterize. wiki/kernels/block-reduce.md.
- [x] variance-reduce:  -> DONE(covered) by kernel-layernorm-hopper (inline sum_x/sum_x2 -> var=E[x^2]-E[x]^2); purpose=both. No separate page needed. online two-pass/Welford variance (for norm); purpose=both. attach to rmsnorm/layernorm.
- [x] argmax-row:  -> DONE speedup; correctness 100% match; 1.10-1.76x torch.argmax. wiki/kernels/argmax.md. per-row argmax (sampling) vs torch; characterize. wiki/kernels/argmax.md.

### QUANT
- [x] per-tensor-w8-quant:  -> DONE speedup; correctness PASS (fp32 exact, bf16 maxdiff<=1); 2.3-3.6x faster than torch at large N. wiki/kernels/per-tensor-quant.md. symmetric per-tensor int8/fp8 quant; purpose=speedup (enables low-prec GEMM). wiki/kernels/per-tensor-quant.md.
- [x] per-channel-quant:  -> DONE both; correctness PASS (maxdiff<=1, scales exact); fused 6.25-12.08x faster than torch. wiki/kernels/per-channel-quant.md. per-channel weight quant; purpose=both (speedup + accuracy). wiki/kernels/per-channel-quant.md.
- [x] fp8-e4m3-quant:  -> DONE speedup; correctness PASS (~99.996% exact); 2.0-3.06x faster than torch at large N. wiki/kernels/fp8-quant.md. fp8 e4m3 quantization kernel (amax->scale->cast); purpose=speedup. wiki/kernels/fp8-quant.md.
- [x] int4-dequant-w4a16:  -> DONE speedup; correctness bit-identical (match=1.0); 19-37x faster than torch. wiki/kernels/int4-dequant.md. INT4 weight dequant to fp16/bf16; purpose=speedup (memory). wiki/kernels/int4-dequant.md.
- [x] fp8-oob-clamp:  -> DONE robustness; clamped kernel 0 NaN (max 448) vs naive 64 NaN on outliers/Inf; clamp overhead ~1.6%. Attached to wiki/kernels/fp8-quant.md + bundle. fp8 quant with OOB clamping (no NaN); purpose=robustness. attach to fp8-quant.

### ATTENTION
- [x] fa2-causal:  -> DONE speedup; correctness PASS (err 3-6e-5); 0.36-0.66x torch SDPA (backend-wins, same as non-causal). Attached causal section to wiki/kernels/triton-fa2-hopper.md + bundle. FA-2 forward with causal mask vs torch SDPA(causal); purpose=speedup. attach to wiki/kernels/triton-fa2-hopper.md.
- [x] gqa-mqa-attn:  -> DONE speedup; correctness PASS (err 3.8-7.6e-6); 0.27-0.47x torch SDPA (backend-wins; GQA mapping correct). wiki/kernels/gqa-mqa-attn.md. grouped/multi-query attention vs torch SDPA; purpose=speedup. attach to triton-fa2-hopper or new.
- [x] sliding-window-attn:  -> DONE both; correctness PASS (err 2.4e-4); 6.9-14.3x naive ref; finite-neg masking avoids NaN; window-capping deferred. wiki/kernels/sliding-window-attn.md. sliding-window causal mask; purpose=speedup. attach to triton-fa2-hopper.

### COPY/LAYOUT
- [x] tiled-transpose-square:  -> DONE speedup; bit-identical; 2.79-3.50x torch (fp32). wiki/kernels/tiled-transpose.md. coalesced tiled transpose vs naive; purpose=speedup. wiki/kernels/tiled-transpose.md.
- [x] swizzle-layout:  -> DEFERRED-to-existing wiki/techniques/swizzling.md (swizzle is a layout/access-reorder technique, not a standalone operator; demonstrated inside tiled transpose / matmul tiling). row->col-major swizzle for matmul layout; purpose=speedup. attach to swizzling or new.

### SCAN/SORT
- [x] prefix-sum-scan:  -> DONE speedup; fp32 correctness PASS (rel 1e-6); 2.9-4.0x moderate N, 0.31-0.35x N=32000 (register spill). wiki/kernels/prefix-scan.md. block + warp-up prefix sum (exclusive); characterize. wiki/kernels/prefix-scan.md.
- [x] topk-selection:  -> DONE speedup; correctness PASS (val+idx exact); 3.54-4.14x moderate N, 0.57-0.58x N=32000. wiki/kernels/topk.md. per-row top-k (sampling) vs torch.topk; characterize. wiki/kernels/topk.md.

### MOE
- [x] moe-topk-gating:  -> DONE speedup; correctness PASS (werr 1e-7, set_match=1.0); 1.29-3.69x torch. wiki/kernels/moe-gating.md. MoE top-k gating + softmax (routing) vs torch; purpose=speedup. wiki/kernels/moe-gating.md.
- [x] moe-permute-unpermute:  -> DONE speedup(record-negative); bit-identical gather+scatter; ~parity torch (0.83-1.06x, memcpy-bound). wiki/kernels/moe-permute.md. scatter/gather permute for MoE dispatch; purpose=speedup. attach to moe-gating or new.

### SPARSE/CONV
- [x] block-sparse-matmul:  -> DONE speedup(record-negative); correctness PASS (err=0); naive random-tile ~0.01x dense (L2 locality destroyed); needs locality-aware tile sorting. wiki/kernels/block-sparse-matmul.md. block-sparse (spMM) vs dense masked; purpose=speedup. wiki/kernels/block-sparse-matmul.md.
- [x] depthwise-conv:  -> DONE speedup; bit-identical; ~parity small / 1.53-1.76x faster large spatial. wiki/kernels/depthwise-conv.md. depthwise conv (im2col or direct); characterize. wiki/kernels/depthwise-conv.md.
