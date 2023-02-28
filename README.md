# MaMiDa-CourseProject-VS

This repository entails the code and term paper for the course "Macro with Micro Data" (Winter semester 2022/2023) by Prof. Dürnecker. It largely replicates the reduced form evidence proposed by Mueller, Spinnewejn and Topa's 2021 Paper ""Job Seekers' Perceptions and Employment Prospects: Heterogeneity, Duration Dependence, and Bias". The projects uses *pytask* by Raabe (2020) as workflow management system following the idea of making it easily accesible and reproducible.

## How to get started

Install the conda environment. The only requirement for this is to have a [Anaconda Distribution](https://www.anaconda.com/products/distribution) available.
Then you have to move to the folder using the command line

```
$ cd your/folder/path
```

Next, you have to install and activate the conda environment. It creates a python environment with all required packages for the replication.

```
$  conda env create -f environment.yml
$  conda activate data_manager
```

Now, we have to invoke pytask by typing

```
$  pip install -e .

$  pytask
```

After this, the code should run between about 10 minutes. The biggest contributors to the compile length are the large .xlsx files that
have to be read in and the code to bootstrap the lower bound of the ex-ante variance in job finding probability.
If all of this ran without error, you should see a **bld** folder containing all figures and tables produced.
Additionally, pytask compiles the latex file and produces the term paper.

## References

- Raabe, T. (2021). *A python tool for managing scientic workflows*. [pytask](https://github.com/pytask-dev/pytask)
- Mueller, A. I., Spinnewijn, J. and Topa, G. (2021). *Job Seekers' Perceptions and Employment Prospects: Heterogeneity, Duration Dependence, and Bias*.
American Economic Review 111.1. p. 324-63.
