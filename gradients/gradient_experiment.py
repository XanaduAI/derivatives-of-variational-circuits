"""Performs a single experiment for a given choice of hyper-parameters."""
import time
import itertools

import numpy as np
import pennylane as qml
import yaml
from pennylane.init import *
from pennylane.templates import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
    amplitude_damping_error,
)
from qiskit import IBMQ

# Set of Ibm backend names
IBM_BACKEND_NAMES = {
    "ibmqx2",
    "ibmq_16_melbourne",
    "ibmq_valencia",
    "ibmq_london",
}


def get_ibm_data(hyperparameters):
    """
    Returns a dictionary containing info about IBM backends.
    To avoid useless loading times, only backends whose names are
    both in IBM_BACKEND_NAMES and hyperparameters["ns_type"] are considered.
    """
    if isinstance(hyperparameters["ns_type"], list):
        backends_ns = IBM_BACKEND_NAMES & set(hyperparameters["ns_type"])
    else:
        backends_ns = IBM_BACKEND_NAMES & {hyperparameters["ns_type"]}

    if not backends_ns:
        return {}
    print("Loading IBMQ account... ", end="")
    IBMQ.load_account()

    if "ibmq_valencia" in backends_ns:
        raise NotImplementedError("Insert your IBMQ details here")
        # provider = IBMQ.get_provider(
        #     hub="...", group="...", project="..."
        # )
        # ibm_data = provider
    elif "ibmq_london" in backends_ns:
        raise NotImplementedError("Insert your IBMQ details here")
        # provider = IBMQ.get_provider(
        #     hub='...', group='...', project='...'
        # )
        # ibm_data = provider
    else:
        provider = IBMQ.get_provider(group="open")
        ibm_data = {}
        for be_name in backends_ns:
            ibm_device = provider.get_backend(be_name)
            device_properties = ibm_device.properties()
            noise_model = NoiseModel.from_backend(device_properties)
            ibm_data.update(
                {
                    be_name: {
                        "noise_model": noise_model,
                    }
                }
            )
    print("Done.")
    return ibm_data


def run(params, ibm_data=None, **kwargs):
    """Runs a single experiment with a given choice of the hyper-parameters."""

    same_seed = params.pop("same_seed", False)
    if same_seed:
        seed_store = params["seed"]

        if params["dev_ns"] != "ibmq":
            params["seed"] = 0
        else:
            params["seed"] = 17

        if params["circ"] == "hw_friendly":
            params["seed"] = 2

    init_time = time.time()
    params = params.copy()
    np.random.seed(params["seed"])

    ##############################################################################
    # Load devices

    dev = load_device(params, exact=False, ibm_data=ibm_data)
    dev_exact = load_device(params)

    ##############################################################################
    # Define ansatz and quantities of interest

    ansatz = load_ansatz(params)

    # symbolic evaluation of expectation values
    exp_exact = qml.QNode(ansatz, dev_exact, diff_method="parameter-shift")
    exp_sampled = qml.QNode(ansatz, dev, diff_method="parameter-shift")

    # symbolic evaluation of gradients
    # note: argnum=0 corresponds to the full "weights" array
    grad_exact = qml.grad(exp_exact, argnum=0)
    grad_shift = qml.grad(exp_sampled, argnum=0)

    if params["dev_ns"] == "ibmq":
        exp_sampled_old = exp_sampled

        def exp_sampled(*args, **kwargs):
            """Wrapped exp_sampled to try again when there are timeouts."""
            finished = False

            while not finished:
                try:
                    res = exp_sampled_old(*args, **kwargs)
                    finished = True
                except:
                    pass

            return res

    ##############################################################################
    # Evaluate quantities
    w = load_weights(params)
    result = {}

    np.random.seed()

    # expectation values
    if params.pop("log_exp", False):
        e_exact = exp_exact(w)
        e_sampled = exp_sampled(w)

        result["exp_e"] = e_exact
        result["exp_s"] = e_sampled

    if params.pop("log_g_ex", False):
        g_exact = grad_exact(w)
        result["g_ex"] = g_exact.tolist()

    if params.pop("log_g_sh", False):
        g_shift = grad_shift(w)
        result["g_sh"] = g_shift.tolist()

    eval_grads(params, w, exp_sampled, result)
    eval_hess(params, w, exp_sampled, result)

    ##############################################################################
    # Output results
    result["time"] = time.strftime("%Y%m%d-%H%M%S")
    result["rtime"] = time.time() - init_time

    if same_seed:
        params["seed"] = seed_store

    return {**params, **result}


def load_device(params: dict, exact: bool = True, ibm_data: dict = {}) -> qml.Device:
    """Load simulator or QPU."""
    if exact:
        if params["dev_ex"] == "def.qub":
            dev = qml.device("default.qubit", wires=params["n_qubits"])
        else:
            raise ValueError("Unknown exact device")
    else:
        # zero_noise default qubit with finite shots
        if params["dev_ns"] == "def.qub":
            dev = qml.device(
                "default.qubit", wires=params["n_qubits"], shots=params["n_shots"], analytic=False,
            )
        elif params["dev_ns"] == "ibmq":
            if ibm_data:
                dev = qml.device(
                    "qiskit.ibmq",
                    wires=params["n_qubits"],
                    backend=params["ns_type"],
                    shots=params["n_shots"],
                    provider=ibm_data,
                )
            else:
                dev = qml.device(
                    "qiskit.ibmq",
                    wires=params["n_qubits"],
                    backend=params["ns_type"],
                    shots=params["n_shots"],
                )
        elif params["dev_ns"] == "qsk.aer":
            # depolarizing noise
            if params["ns_type"] == "dep":
                noise_model = NoiseModel()
                noise_model.add_all_qubit_quantum_error(
                    depolarizing_error(params["noise"], 1), ["u1", "u2", "u3"]
                )
            # amplitude damping noise
            elif params["ns_type"] == "amp":
                noise_model = NoiseModel()
                noise_model.add_all_qubit_quantum_error(
                    amplitude_damping_error(params["noise"], 1), ["u1", "u2", "u3"]
                )
            # noise model from device
            elif params["ns_type"] in ibm_data.keys():
                dev_data = ibm_data[params["ns_type"]]
                noise_model = dev_data["noise_model"]
            # zero_noise (i.e. exact "qsk.aer" simulator)
            elif params["ns_type"] == "zero_noise":
                noise_model = None
            else:
                raise ValueError("Unknown noise source")

            dev = qml.device(
                "qiskit.aer",
                wires=params["n_qubits"],
                shots=params["n_shots"],
                noise_model=noise_model,
                backend="qasm_simulator",
            )
        else:
            raise ValueError("Unknown noisy device")

    return dev


def load_ansatz(params: dict):
    """Load ansatz circuit."""
    if params["circ"] == "SEL":
        resizer = lambda x: [x.reshape(params["depth"], params["n_qubits"], 3)]
        circ = StronglyEntanglingLayers
    elif params["circ"] == "BEL":
        resizer = lambda x: [x.reshape(params["depth"], params["n_qubits"])]
        circ = BasicEntanglerLayers
    elif params["circ"] == "hw_friendly":
        if params["n_qubits"] > 5:
            raise ValueError("Must use at most 5 qubits with hardware friendly layer")
        if params["meas"] != "PZ1":
            raise ValueError("Must measure the first qubit when using hardware friendly layer")

        resizer = lambda x: [x.reshape(params["depth"], params["n_qubits"])]

        def layer(weights, layer_number):
            for i in range(params["n_qubits"]):
                qml.RX(weights[layer_number, i], wires=i)

            if params["n_qubits"] > 1:
                qml.CNOT(wires=[0, 1])
                if params["n_qubits"] > 2:
                    qml.CNOT(wires=[2, 1])
                    if params["n_qubits"] > 3:
                        qml.CNOT(wires=[3, 1])
                        if params["n_qubits"] > 4:
                            qml.CNOT(wires=[4, 3])

        def circ(weights, wires):
            for j in range(params["depth"]):
                layer(weights, j)

    elif params["circ"] == "SEL1":
        if params["init_method"] == "uniform":
            other_weights = strong_ent_layers_uniform(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            )
        elif params["init_method"] == "normal":
            other_weights = strong_ent_layers_normal(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            )

        resizer = lambda x: [np.array([other_weights[0, 0, 0], x[0], other_weights[0, 0, 2]])]
        other_weights[0, 0, :] = 0 * other_weights[0, 0, :]

        def circ(weights, wires):
            qml.Rot(*weights, wires=0)
            StronglyEntanglingLayers(other_weights, wires=wires)

    elif params["circ"] == "RL":
        resizer = lambda x: [x.reshape(params["depth"], params["n_qubits"])]
        circ = RandomLayers
    elif params["circ"] == "S2D":

        def resizer(x):
            w_init = x[: params["n_qubits"]]
            w_main = x[params["n_qubits"] :].reshape(params["depth"], params["n_qubits"] - 1, 2)
            return [w_init, w_main]

        if params["n_qubits"] > 1:
            circ = SimplifiedTwoDesign
        else:

            def circ(w_init, w_main, wires):
                qml.RY(w_init[0], wires=wires)

    else:
        raise ValueError("Unknown circuit")

    def ansatz(weights: np.ndarray):
        """Ansatz for variational circuit"""
        weights = resizer(weights)
        circ(*weights, wires=list(range(params["n_qubits"])))

        if params["meas"] == "PZ0":
            return qml.expval(qml.PauliZ(0))
        if params["meas"] == "PZA":
            obs = qml.PauliZ(0)

            for i in range(1, params["n_qubits"]):
                obs = obs @ qml.PauliZ(i)

            return qml.expval(obs)
        if params["meas"] == "PZ1":
            return qml.expval(qml.PauliZ(1))
        else:
            raise ValueError("Unknown measurement")

    return ansatz


def load_weights(params: dict) -> np.ndarray:
    """Initializes the weights of the quantum circuit."""
    method = params["init_method"]

    if params["circ"] == "SEL":
        if method == "uniform":
            w = strong_ent_layers_uniform(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        elif method == "normal":
            w = strong_ent_layers_normal(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        else:
            raise ValueError("Incorrect method")

    elif params["circ"] == "hw_friendly":
        if method == "uniform":
            w = basic_entangler_layers_uniform(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        elif method == "normal":
            w = basic_entangler_layers_normal(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        else:
            raise ValueError("Incorrect method")

    elif params["circ"] == "BEL":
        if method == "uniform":
            w = basic_entangler_layers_uniform(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        elif method == "normal":
            w = basic_entangler_layers_normal(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        else:
            raise ValueError("Incorrect method")

    elif params["circ"] == "SEL1":
        if method == "uniform":
            w = basic_entangler_layers_uniform(
                n_wires=1, n_layers=1, seed=params["seed"]
            ).flatten()
        elif method == "normal":
            w = basic_entangler_layers_normal(
                n_wires=1, n_layers=1, seed=params["seed"]
            ).flatten()
        else:
            raise ValueError("Incorrect method")

    elif params["circ"] == "RL":
        if method == "uniform":
            w = random_layers_uniform(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        elif method == "normal":
            w = random_layers_normal(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        else:
            raise ValueError("Incorrect method")

    elif params["circ"] == "S2D":
        if method == "uniform":
            w_init = simplified_two_design_initial_layer_uniform(
                n_wires=params["n_qubits"], seed=params["seed"]
            ).flatten()
            w_main = simplified_two_design_weights_uniform(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        elif method == "normal":
            w_init = simplified_two_design_initial_layer_normal(
                n_wires=params["n_qubits"], seed=params["seed"]
            ).flatten()
            w_main = simplified_two_design_weights_normal(
                n_wires=params["n_qubits"], n_layers=params["depth"], seed=params["seed"]
            ).flatten()
        else:
            raise ValueError("Incorrect method")

        w = np.array(list(w_init) + list(w_main))

    else:
        raise ValueError("Incorrect circuit")

    return w


def eval_hess(params: dict, w: np.ndarray, exp_samples, result: dict):
    """Evaluates the hessian using parameter shift and finite difference"""
    if params.pop("log_hess", False):
        s = params["h"]
        shift = np.eye(len(w))
        denom = 4 * np.sin(s) ** 2
        denom_fd = 4 * s ** 2
        weights = w.copy()
        hess = np.zeros((len(weights), len(weights)))

        for c in itertools.combinations_with_replacement(range(len(weights)), r=2):
            weights_pp = weights + s * (shift[c[0]] + shift[c[1]])
            weights_pm = weights + s * (shift[c[0]] - shift[c[1]])
            weights_mp = weights - s * (shift[c[0]] - shift[c[1]])
            weights_mm = weights - s * (shift[c[0]] + shift[c[1]])

            f_pp = exp_samples(weights_pp)
            f_pm = exp_samples(weights_pm)
            f_mp = exp_samples(weights_mp)
            f_mm = exp_samples(weights_mm)

            num = f_pp - f_mp - f_pm + f_mm
            hess[c] = num

        hess = hess + hess.T
        for i in range(params["n_qubits"]):
            hess[i, i] /= 2

        result["hess_ps"] = hess / denom
        result["hess_fd"] = hess / denom_fd  # https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm


def eval_grads(params: dict, w: np.ndarray, exp_sampled, result: dict):
    """Evaluates the finite difference, partial shift, and max shift gradients."""
    basis = np.eye(len(w))
    g_fd = []
    g_partial_shift = []
    g_max_shift = []

    exp_shift_h_a = []
    exp_shift_h_b = []
    exp_shift_max_a = []
    exp_shift_max_b = []

    fd_flag = params.pop("log_g_fd", False)
    ps_flag = params.pop("log_g_ps", False)
    ms_flag = params.pop("log_g_ms", False)

    for shift in basis:
        if fd_flag or ps_flag:
            # finite difference method
            shift_a = w + params["h"] * shift
            shift_b = w - params["h"] * shift
            _exp_shift_h_a = exp_sampled(shift_a)
            _exp_shift_h_b = exp_sampled(shift_b)
            diff = _exp_shift_h_a - _exp_shift_h_b
            _grad_fd = diff / (2 * params["h"])

            exp_shift_h_a.append(_exp_shift_h_a)
            exp_shift_h_b.append(_exp_shift_h_b)

            # partial-shift method
            _grad_partial_shift = diff / (2 * np.sin(params["h"]))

            g_fd.append(_grad_fd)
            g_partial_shift.append(_grad_partial_shift)

        if ms_flag:
            # standard parameter-shift method
            h_max = np.pi / 2
            shift_a = w + h_max * shift
            shift_b = w - h_max * shift
            _exp_shift_max_a = exp_sampled(shift_a)
            _exp_shift_max_b = exp_sampled(shift_b)
            diff = _exp_shift_max_a - _exp_shift_max_b
            _grad_max_shift = diff / 2

            exp_shift_max_a.append(_exp_shift_max_a)
            exp_shift_max_b.append(_exp_shift_max_b)

            g_max_shift.append(_grad_max_shift)

    if fd_flag:
        result["g_fd"] = [float(x) for x in g_fd]
    if ps_flag:
        result["g_ps"] = [float(x) for x in g_partial_shift]
    if ms_flag:
        result["g_ms"] = [float(x) for x in g_max_shift]

    if params.pop("log_fd_shift", False):
        result["fd_s_a"] = [float(x) for x in exp_shift_h_a]
        result["fd_s_b"] = [float(x) for x in exp_shift_h_b]
    if params.pop("log_mx_shift", False):
        result["mx_s_a"] = [float(x) for x in exp_shift_max_a]
        result["mx_s_b"] = [float(x) for x in exp_shift_max_b]


if __name__ == "__main__":

    hyper_params = {
        "n_qubits": 2,
        "depth": 3,
        "n_shots": 10 ** 4,
        "noise": 0.01,
        "h": 0.1,
        "seed": 42,
        "init_method": "uniform",
        "dev_ns": "qsk.aer",
        "ns_type": "dep",
        "dev_ex": "def.qub",
        "circ": "SEL",
        "meas": "PZ0",
        "log_g_sh": True,
        "log_g_fd": True,
        "log_g_ex": True,
        "log_g_ms": True,
        "log_g_ps": True,
        "log_fd_shift": True,
        "log_mx_shift": True,
    }

    _ibm_data = get_ibm_data(hyperparameters=hyper_params)
    r = run(hyper_params, ibm_data=_ibm_data)

    print("Printing from __main__ :", yaml.dump(r, default_flow_style=False))
