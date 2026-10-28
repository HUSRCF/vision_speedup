# Omitted world initial FFN on adopted219

This diagnostic preserves actual adopted219 spatial/temporal/native-attention
routing in both paired conditions. Only the candidate halves rows in the original
`_bk32_native_hidden_parts` and `_prepacked_gemm` JIT launches, keeping the full
Python owner shape40960x512, weight-version caches, BK32, tile options and exact
K accumulation. One whole-FFN output gather sends40MiB/rank; no inner gather.
Both halves align to complete GROUP32*BM64 boundaries. No new JIT body.

CPU checks verify actual storage offsets and all unchanged nonrow arguments.
The clean GPU1/2 complete world test passes six dynamic states/18 signedzero
outputs perrank,570 Linear/782 norm/192 Conv2d masters, RNG and baseline
restoration, and original23+45 native owner/CUBIN gates. All six timing pairs
are faster, with median savings3.940857/3.897850ms including gather. Do not add
rank savings. Full V2/PSNR remains pending; no official score is inferred.

Only M20480 variants use a fresh perrank standard Triton cache on an executable
home mount, restoring the original warm cache immediately. No custom manager,
old-cache deletion, input/output replay, new precision or kernel-parameter
export. Dynamic master refresh is handled by original world initial owner.

The first attempt terminated on a diagnostic KeyError for the previous report
schema after graph construction. R2 changes only the coverage assertion; the
operator helper has the same SHA256. No performance/quality result from R1 is
used. Stage scripts in a fresh private directory; original209 remains model
package source and ADOPTED219_ROOT supplies its verified10 helper files.
Run the CPU fixture with CUDA hidden, then the guarded model retry entry with
normal original209 flags, ADOPTED219_ROOT and INIT_FFN_CACHE_DIR. This is a
model diagnostic, not a timed submission.

Follow-up: the exact operator is now qualified in production220. Complete
paired V2 was278.896881->277.436969s, PSNR29.675324786dB and all23 video hashes
unchanged; fresh official103 validation passed. The model summary's pending
flags describe its frozen pre-formal snapshot. See ../production220.
