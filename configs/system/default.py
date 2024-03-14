num_workers = 4

# common configs
mixed_precision = "fp16"
grad_checkpoint = True

# if using zero only
plugin = "zero2"
sequence_parallelism = False
sp_size = 1

# if use sequence parallelism + zero
# plugin = "zero2-seq"
# sequence_parallelism = True
# sp_size = 2
