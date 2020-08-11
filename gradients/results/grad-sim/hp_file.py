"""Hyperparameters script for experiment"""
import numpy as np

seed = list(range(1000))
n_qubits = [5]
n_shots = [10, 14, 20, 29, 41, 59, 84, 100, 119, 170, 242, 346, 492, 702, 1000, 1425, 2031, 2894,
           4125, 5878, 8192, 8377, 11938, 17013, 24245, 34551, 49239, 70170, 100000]
depth = [1]
noise = [0.0]  # It shouldn't do anything
h = [0.05, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.18, 0.2 , 0.22, 0.24,
       0.26, 0.28, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.44, 0.45,
       0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67,
       0.69, 0.71, 0.73, 0.75, 0.77, 0.8 , 0.82, 0.83, 0.84, 0.86, 0.88,
       0.9 , 0.92, 0.94, 0.96, 0.98, 1.  , 1.02, 1.04, 1.06, 1.08, 1.1 ,
       1.12, 1.14, 1.16, 1.18, 1.2 , 1.22, 1.22, 1.24, 1.27, 1.29, 1.31,
       1.33, 1.35, 1.37, 1.39, 1.41, 1.43, 1.45, 1.47, 1.49, 1.51, 1.53,
       1.55, 1.57, 1.59, 1.61, 1.63, 1.65, 1.67, 1.69, 1.71, 1.73,
       1.76, 1.78, 1.8 , 1.82, 1.84, 1.86, 1.88, 1.9 , 1.92, 1.94, 1.96,
       1.98, 2.  ]
init_method = ["uniform"]
dev_ns = ["def.qub"]
ns_type = ["none"]
dev_ex = ["def.qub"]
circ = ["hw_friendly"]
meas = ["PZ1"]

log_g_sh = False  # Logging this increases number of experiments, can instead include pi/2 in h
log_g_fd = True
log_g_ex = True
log_g_ms = False  # Logging this increases number of experiments
log_g_ps = True
log_fd_shift = False
log_mx_shift = False
same_seed = True

dask = True

hyperparameters = {
    "seed": seed,
    "n_qubits": n_qubits,
    "n_shots": n_shots,
    "depth": depth,
    "noise": noise,
    "h": h,
    "init_method": init_method,
    "dev_ns": dev_ns,
    "ns_type": ns_type,
    "dev_ex": dev_ex,
    "circ": circ,
    "meas": meas,
}

log_parameters = {
    "log_g_sh": log_g_sh,
    "log_g_fd": log_g_fd,
    "log_g_ex": log_g_ex,
    "log_g_ms": log_g_ms,
    "log_g_ps": log_g_ps,
    "log_fd_shift": log_fd_shift,
    "log_mx_shift": log_mx_shift,
    "same_seed": same_seed,
}
