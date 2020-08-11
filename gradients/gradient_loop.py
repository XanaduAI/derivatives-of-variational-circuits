"""
Loops over experiments specified in a user-input directory in ./results/ which contains a
``hp_file.py`` script to fix the hyperparameters.
"""
import hashlib
import itertools
import pickle
import sys
import dask

import gradient_experiment as exp


def run(name, hp, ibm_data):
    """Runs benchmarking for the given values."""

    result = exp.run(hp, ibm_data=ibm_data)

    for h in hp_not_looped:  # removes hyperparameters from dictionary that are not looped over
        del result[h]

    return {name: result}


def make_hash(hp: dict):
    """Makes hash from hyperparameters."""
    v_l = list(hp.values())
    for j, v in enumerate(v_l):
        if v == True:
            v_l[j] = "1"
        elif v == False:
            v_l[j] = "0"
        else:
            v_l[j] = str(v)

    v_l = "".join(v_l)

    return hashlib.md5(v_l.encode("utf-8")).hexdigest()


if __name__ == '__main__':

    try:
        exp_name = sys.argv[1]  # Name of folder containing experiment
    except IndexError:
        raise IndexError("Must input name of experiment and specify whether to log")

    try:
        SAVE_REPS = int(sys.argv[2])  # How often to save
    except IndexError:
        SAVE_REPS = 100

    exp_dir = "./results/" + exp_name + "/"

    try:
        sys.path.append(exp_dir)
        import hp_file # sample script can be found in working directory
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Cannot find hyperparameters script in folder {}".format(exp_dir))

    try:
        multicore = hp_file.dask
    except:
        multicore = True

    combinations = list(itertools.product(*hp_file.hyperparameters.values()))
    print("Number of combinations: {}".format(len(combinations)))

    filename = exp_dir + "results.pickle"

    try:
        with open(filename, "rb") as file:
            results = pickle.load(file)
    except FileNotFoundError:
        results = {}

    # Get IBM data only once before the loop
    ibm_data = exp.get_ibm_data(hp_file.hyperparameters)

    # hyperparameters that are not looped over in experiment
    hp_not_looped = [k for k, v in hp_file.hyperparameters.items() if len(v) == 1]

    r = []

    for i, values in enumerate(itertools.product(*hp_file.hyperparameters.values())):
        hp = {**dict(zip(hp_file.hyperparameters.keys(), values)), **hp_file.log_parameters}
        name = make_hash(hp)

        if multicore:
            if name not in results:
                r.append(dask.delayed(run)(name, hp, ibm_data))
        else:
            if name not in results:
                r.append(run(name, hp, ibm_data))

        if len(r) == SAVE_REPS or i == len(combinations) - 1:
            if multicore:
                r = dask.compute(*r, scheduler='processes')
            for d in r:
                results.update(d)

            r = []

            with open(filename, "wb") as file:
                pickle.dump(results, file)
            print("Saved on iteration {}".format(i))
