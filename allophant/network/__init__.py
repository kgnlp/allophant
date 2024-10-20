from torch.backends import cuda, cudnn


# Enable TensorFloat-32 for Ampere or newer GPUs for substantial speed improvements
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
