import gc
import psutil
import time
import torch


def get_ram():
    mem = psutil.virtual_memory()
    free = mem.available / 1024 ** 3
    total = mem.total / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return total - free, f'RAM     : {total - free:.2f}/{total:.2f}GB\t\t RAM      [' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'


def get_vram():
    # free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    occupied = torch.cuda.memory_allocated() / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    free = total - occupied
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'VRAM    : {total - free:.2f}/{total:.2f}GB\t\t VRAM     [' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def get_max_vram():
    max_vram = torch.cuda.max_memory_allocated("cuda") / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * (total - max_vram) / total)
    return max_vram, f'VRAM max: {max_vram:.2f}/{total:.2f}GB\t\t VRAM Max [' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def check_isnan(t, step=""):
    if t.isnan().any():
        print(f"Step {step}: Nan numbers {torch.sum(t.isnan())}")

class CustomLogger:

    def __init__(self, debug=True):
        self.time = time.time()
        self.start_time = self.time
        self.max_vram = 0
        self.max_ram = 0
        if debug:
            print("Starting record max and time")

    def print(self, step="None", check=[], debug=True):
        t = time.time()
        if debug:
            print(f"[{step}]: {t - self.time:.2f}s elapsed")
        self.time = t

        max_vram, max_vram_message = get_max_vram()
        self.max_vram = max(self.max_vram, max_vram)

        gc.collect()
        torch.cuda.empty_cache()

        max_ram, ram_message = get_ram()
        self.max_ram = max(self.max_ram, max_ram)

        vram_message = get_vram()
        if debug:
            print(f"[{step}]: {ram_message}")
            print(f"[{step}]: {vram_message}")
            print(f"[{step}]: {max_vram_message}")

        torch.cuda.reset_peak_memory_stats("cuda")

        for tensor in check:
            if isinstance(tensor, list):
                for t in tensor:
                    check_isnan(t, step)
                if debug:
                    print(tensor[0].shape)
            else:
                check_isnan(tensor, step)
                if debug:
                    print(tensor.shape)


    def summary(self, debug=True):
        if debug:
            print("End")
            print(f"Fake max RAM: {self.max_ram}")
            print(f"Real max VRAM: {self.max_vram}")
            print(f"Total time: {time.time() - self.start_time}")
