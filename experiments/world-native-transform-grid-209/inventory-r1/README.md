# Native Winograd transform geometry: observation only

Fresh synthetic convolutions,14 original geometries ×2 layouts ×N16/N8:56
native CUDA captures matched their corresponding eager outputs bitwise on
a clean locked GPU3 attempt. No graph/kernel/parameter/grid edits or new JIT.
Only names,grids,blocks,shared bytes,argument sizes and aggregate comparisons
are exported; no argument bytes/addresses, weights or operands.

There was no directly halving transform grid axis in these observations. Some
N8 calls select another cuDNN algorithm; others retain the same native grid.
Thus this inventory does not justify halving a transform grid or prove any
reduced-grid numerical/performance result.
[Evidence](../../../results/aggressive-20261028/world-native-transform-grid-209/inventory-r1).

Use original209 on PYTHONPATH in a fresh private scratch directory, set
CANONICAL_ROOT,EXPERIMENT_ROOT,CUDA_VISIBLE_DEVICES and MONITORED_GPU, then run
guard_world_transform209.py --gpu N --entry audit_world_transform_grid209.py.
The guard must acquire both locks and recheck strict idleness before allocation.
