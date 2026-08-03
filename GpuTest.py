import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # Print the name of the first GPU
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")
        
        # Create a tensor and move it to the GPU
        cpu_tensor = torch.randn(3, 3)
        gpu_tensor = cpu_tensor.to('cuda')
        print(f"\nTensor created on CPU:\n{cpu_tensor}")
        print(f"Tensor moved to GPU:\n{gpu_tensor}")
        print(f"Device of GPU tensor: {gpu_tensor.device}")
        
        # Perform a simple operation on the GPU
        result_gpu = gpu_tensor * 2
        print(f"Result of operation on GPU:\n{result_gpu}")
    
else:
    print("CUDA is not available. PyTorch will run on CPU.")
    
    # Create a tensor on the CPU
    cpu_tensor = torch.randn(3, 3)
    print(f"\nTensor created on CPU:\n{cpu_tensor}")
    print(f"Device of CPU tensor: {cpu_tensor.device}")