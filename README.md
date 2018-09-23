# A flexible temporal velocity model for fast contaminant
<!-- transport simulations in porous media # -->

<!-- ### What is this repository for? ### -->

This repository is for modeling macro-dispersion in continuum scale porous media
with Gaussian and exponential correlation structures using the discrete time Markov velocity process (DTMVP).


<!-- For details about the ideas and models used in this code please refer to -->
<!-- [Temporal Markov Processes for Transport in Porous Media: Random Lattice Networks](https://arxiv.org/abs/1708.04173) -->
### Setup ###

* First clone this repository.
* Add the parent directory of py_dp to your Python path. In linux you will need to
add a line similar to `export PYTHONPATH=$PYTHONPATH:/PATH_TO_PARENT_OF_py_dp` to your `~/.bashrc` file.
* All dependencies for this repository are listed in the `environment.yml` file in the root directory.
Using Conda, you can create a new environment will all required dependencies by running
`conda env create -f environment.yml
`
<!-- Make sure you have Python2.7 along with Matplotlib, Numpy, Scipy, Cython and pyamg. -->
<!-- All of these packages can be installed using pip. (e.g. `pip install numpy`). -->
* Complete Cython installations:
    - go to the dispersion directory: `cd py_dp/dispersion/`
    - compile the cython files using these commands:
        * `python setup_count.py build_ext --inplace`
        * `python setup_convert.py build_ext --inplace`

### Getting started with the code ###

After the setup steps you can run `sample_scripts/workflow_dispersion_in_continua.py`.
This file includes all the steps necessary to generate Monte Carlo simulations and
extract the DTMVP model for a small test case.


