"""Hyperparameters script for experiment"""
import numpy as np

seed = list(range(1000))
n_qubits = [5]
n_shots = [1000]
depth = [1]
noise = [0.0]  # It shouldn't do anything
h = [0.05, 0.15, 0.44, 0.83, 1.22, 1.57, 2.0]
init_method = ["uniform"]
dev_ns = ["ibmq"]
ns_type = ["ibmq_valencia"]
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

dask = False

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
