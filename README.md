The python script `check_aquisition_function.py` was used to check the function for PrefMPoI. It can be run from the command line as follows:

```bash
python check_aquisition_function.py
```

To run a full Bayesian optimisation run, the following installations are suggested:

```bash
# The following installation sequence is recommended to run the optimisation:
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/py3
source ~/.venvs/py3/bin/activate
python -m pip install deap==1.3.3
python -m pip install --force-reinstall "numpy==1.13.3"
python -m pip install --force-reinstall "scipy==1.0.0"
python -m pip install "matplotlib==2.1.2"
python -m pip install "pyDOE==0.3.8"
python -m pip install "evoalgos==1.0"
python -m pip install "Cython<3" six paramz
python -m pip install "GPy==1.9.8"
printf "backend: Agg\n" > ~/.config/matplotlib/matplotlibrc
python -m pip install "cma==2.7.0"
python -m pip install "ipython==7.16.3"
python -m pip install "PyFoam==0.6.6"
python -m pip install "numpy-stl==2.16.3"

# Within the IscaOpt directory:
cd FrontCalc
python setup.py install

# Adjust the .sh scripts in each directory for your local cluster
```
