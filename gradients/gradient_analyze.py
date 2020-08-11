"""Functions for analyzing output results dictionary"""
import itertools

import numpy as np

from gradient_loop import make_hash


def calculate_new_quantity(in_keys: list, out_name: str, f, results: dict, hp_file):
    """Calculate new quantity for each data point.

    Calculates the function ``f`` applied to the input ``keys`` and creates a new entry in the
    dictionary with name ``q_name``.
    """
    for i, values in enumerate(itertools.product(*hp_file.hyperparameters.values())):
        hp = {**dict(zip(hp_file.hyperparameters.keys(), values)), **hp_file.log_parameters}
        name = make_hash(hp)

        try:
            r = results[name]
        except KeyError:
            continue

        in_q = [r[k] for k in in_keys]
        out_q = f(*in_q)
        results[name][out_name] = out_q


def avg_quantities(keys: list, results: dict, hp_file):
    """Average quantities specified in ``keys`` over the range of seed values."""
    hps = hp_file.hyperparameters.copy()  # Load hyperparameters
    seed = hps["seed"]  # Record seed list
    del hps["seed"]  # Remove seed from hyperparameters

    results_avg = {}  # Create new dictionary to hold averaged quantities

    for i, values in enumerate(itertools.product(*hps.values())):
        hps_no_seed = dict(zip(hps.keys(), values))
        hps_no_seed_plus_log = {
            **hps_no_seed,
            **hp_file.log_parameters,
        }  # Add back in log parameters needed for hash

        q = []

        for s in seed:
            hp_with_seed = {**{"seed": s}, **hps_no_seed_plus_log}  # Add back in seed within loop
            name = make_hash(hp_with_seed)  # Find corresponding hash
            try:
                q.append([results[name][k] for k in keys])
            except KeyError:
                continue

        hp_not_looped = [
            k for k, v in hps.items() if len(v) == 1
        ]  # Find hyperparameters not looped over
        for h in hp_not_looped:  # Remove hyperparameters from dictionary that are not looped over
            del hps_no_seed[h]

        q = np.array(q)
        q_mean = np.mean(q, axis=0)

        results_avg[make_hash(hps_no_seed)] = {
            **hps_no_seed,
            **{k: q_mean[i] for i, k in enumerate(keys)},
        }

    return results_avg


def access_quantities(keys: list, results: dict, hp_file, average: bool = False):
    """Access quantities specified in ``keys`` over the range of seed values."""
    hps = hp_file.hyperparameters.copy()  # Load hyperparameters
    seed = hps["seed"]  # Record seed list
    del hps["seed"]  # Remove seed from hyperparameters

    results_avg = {}  # Create new dictionary to hold averaged quantities

    for i, values in enumerate(itertools.product(*hps.values())):
        hps_no_seed = dict(zip(hps.keys(), values))
        hps_no_seed_plus_log = {
            **hps_no_seed,
            **hp_file.log_parameters,
        }  # Add back in log parameters needed for hash

        q = []

        for s in seed:
            hp_with_seed = {**{"seed": s}, **hps_no_seed_plus_log}  # Add back in seed within loop
            name = make_hash(hp_with_seed)  # Find corresponding hash
            try:
                q.append([results[name][k] for k in keys])
            except KeyError:
                continue

        hp_not_looped = [
            k for k, v in hps.items() if len(v) == 1
        ]  # Find hyperparameters not looped over
        for h in hp_not_looped:  # Remove hyperparameters from dictionary that are not looped over
            del hps_no_seed[h]

        q = np.array(q)
        if average:
            q = np.mean(q, axis=0)

        results_avg[make_hash(hps_no_seed)] = {
            **hps_no_seed,
            **{k: q[:, i] for i, k in enumerate(keys)},
        }

    return results_avg


def calculate_slice(pinned: dict, results: dict):
    """Extracts all data points from ``results`` with pinned values specified by ``pinned``."""
    slice_d = {}

    for k, v in results.items():
        v_c = v.copy()
        contained = True

        for k_p, v_p in pinned.items():
            if v_c[k_p] != v_p:
                contained = False
            else:
                del v_c[k_p]

        if contained:
            slice_d.update({k: v_c})

    return slice_d


def make_numpy(results_slice: dict, x: str, y: str):
    """Make a slice into two numpy arrays suitable for plotting, with the x and y axes specified
    as key strings."""
    slice_list = sorted([(v[x], v[y]) for k, v in results_slice.items()])
    return np.array([s_l[0] for s_l in slice_list]), np.array([s_l[1] for s_l in slice_list])


def make_numpy2d(results_slice: dict, x: str, y: str, z: str, z_elem: int = None):
    """Make a slice into three numpy arrays suitable for 2D plotting, with the x, y, and z axes
    specified as key strings."""
    slice_list = sorted([(v[x], v[y], v[z]) for k, v in results_slice.items()])
    if z_elem:
        return (
            np.array([s_l[0] for s_l in slice_list]),
            np.array([s_l[1] for s_l in slice_list]),
            np.array([s_l[2][z_elem] for s_l in slice_list]),
        )

    return (
        np.array([s_l[0] for s_l in slice_list]),
        np.array([s_l[1] for s_l in slice_list]),
        np.array([s_l[2] for s_l in slice_list]),
    )
