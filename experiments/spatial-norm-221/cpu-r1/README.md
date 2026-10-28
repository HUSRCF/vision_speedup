# Spatial norm routing draft — CPU only

72 real CPU shape/stride/view-offset checks passed;16 temporal GroupNorm rejection cases passed. No CUDA context initialized. This is not a GPU numerical, speed or video-quality qualification and is not in the production221 manifest or default runner.

The draft wraps the existing spatial router, acting only when scope is exactly the active SpatialTransformer scope. It halves independent N16 sample/group rows for original partial/stats/apply/serial_norm and norm/layout pack. All scalar dimensions and strides, including full M, remain unchanged. It slices sample pointers, stats tables and output positions and halves only the row grid. No collective is added. Temporal5D GroupNorm and temporal routing are explicitly outside this wrapper. Original caches/affines and norm statistics arithmetic remain unchanged.

To run the CPU check with authorized source dependencies, set ADOPTED221_ROOT to the frozen materialized221 root and PYTHONPATH to this folder plus that root/src. Import spatial_norm221_mode before configure when constructing any future model probe. Until paired GPU checks pass, no performance or precision equivalence claim is made.
