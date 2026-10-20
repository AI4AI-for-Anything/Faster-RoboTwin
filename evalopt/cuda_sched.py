"""EVALOPT_CUDA_BLOCKING_SYNC=1: make host threads that wait on this process's CUDA primary context
block instead of spin (CU_CTX_SCHED_BLOCKING_SYNC). It changes only how the CPU waits for the GPU,
not what the GPU computes. Must be imported before anything creates the CUDA context (torch, warp,
the SAPIEN renderer's OIDN denoiser). Import for its side effect; STATUS records what happened.
"""
import ctypes
import os

STATUS = {"requested": os.environ.get("EVALOPT_CUDA_BLOCKING_SYNC") == "1"}
if STATUS["requested"]:
    cu = ctypes.CDLL("libcuda.so.1")
    STATUS["cuInit"] = cu.cuInit(0)
    n = ctypes.c_int()
    cu.cuDeviceGetCount(ctypes.byref(n))
    setf = getattr(cu, "cuDevicePrimaryCtxSetFlags_v2", None) or cu.cuDevicePrimaryCtxSetFlags
    STATUS["set"] = []
    for i in range(n.value):
        dev = ctypes.c_int()
        cu.cuDeviceGet(ctypes.byref(dev), i)
        STATUS["set"].append(setf(dev, 0x04))  # CU_CTX_SCHED_BLOCKING_SYNC
    flags, active = ctypes.c_uint(), ctypes.c_int()
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))
    STATUS["flags_dev0"] = flags.value
