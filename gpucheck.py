import pynvml

def get_gpu_utilization():
    # Initialize NVML
    pynvml.nvmlInit()

    # Get the number of GPUs
    device_count = pynvml.nvmlDeviceGetCount()

    utilization_data = {}
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization_data[f"GPU {i} ({name})"] = {
            "GPU Utilization (%)": utilization.gpu,
            "Memory Utilization (%)": utilization.memory
        }

    # Shutdown NVML
    pynvml.nvmlShutdown()

    return utilization_data

if __name__ == "__main__":
    gpu_utilization = get_gpu_utilization()
    for gpu, data in gpu_utilization.items():
        print(f"{gpu}:")
        for key, value in data.items():
            print(f"  {key}: {value}%")

