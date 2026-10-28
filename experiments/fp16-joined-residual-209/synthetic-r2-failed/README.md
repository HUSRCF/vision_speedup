# Rejected before arithmetic: incompatible cat API

The frozen candidate attempted a two-dimensional ordered cat. The installed
Triton2.3.1 cat signature is(input,other,can_reorder=False); its implementation
accepts only1D blocks and requires can_reorder=True. Compilation stopped with
multiple values for can_reorder. No numerical/timing result is valid.
A subsequent revision tests only the supported chained-dot route.
