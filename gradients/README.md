<img align="middle" src="https://github.com/XanaduAI/derivatives-of-variational-circuits/blob/master/gradients/results/grad-sim/fd-vs-ps-simulator.png" width=400px>

# Numerics for statistical estimation of gradients

This folder contains the source code for generating the plots in the paper focused on statistical
estimation of derivatives.

An experiment is set up by creating a new folder in ``./results/`` and adding a ``hp_file.py`` to
describe the experiment hyperparameters. The ``gradient_loop.py`` script can then be run to
generate numerics for the experiment. Suppose we want to generate numerics for the ``grad-sim``
experiment, this can be done with:

```bash
python gradient_loop.py grad-sim 1000
```

Where ``1000`` indicates the number of iterations between each save of the output data.

# Numerics for statistical estimation of the Hessian

The numerics for the Hessian can be found in the ``hessian.ipynb``, ``hessian_sim.ipynb``, and
``hessian_analysis.ipynb`` notebooks.
