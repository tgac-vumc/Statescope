<p align="center">
  <img width="424" height="200" src="https://github.com/tgac-vumc/Statescope/blob/master/img/Logo_Statescope.png">
</p>

# Statescope
Statescope is a computational framework designed to discover cell states from cell type-specific gene expression profiles inferred from bulk RNA profiles.

<p align="center">
  <img width="75%" height="75%" src="https://github.com/tgac-vumc/Statescope/blob/master/img/Statescope_Overview.png">
</p>


## Installation

### Using pip

The python package of Statescope is available on pip:

```
pip install Statescope
```

We tested Statescope with `python => 3.10`.

### Using conda

In case the pip installation gives you problems, you can also use the environment.yaml file provided on this Github.

```
conda env create -f environment.yaml
conda activate Statescope
```

This install Statescope version 1.0.7 with python 3.10.19

##  Running a demo script

### Basic Tutorial
You can find an basic demo script under the `tutorial` folder.
You can open the script using the command below after installing Statecope:

```
jupyter notebook tutorial/BasicTutorial.ipynb
```

### Advanced Tutorial
You can also find a more advanced demo script under the `tutorial` folder.

```
jupyter notebook tutorial/AdvancedTutorial.ipynb
```

## Documentation

### Website

The documentation of Statescope is hosted on https://tgac-vumc.github.io/Statescopeweb/


## System Requirements

### Hardware Requirements

Statescope can run on the minimal computer spec, such as Binder (1 CPU, 2GB RAM on Google Cloud), when data size is small. However, Statescope can significantly benefit from the larger amount of CPUs and RAM and can leverage GPU hardware.

### OS Requirements

The package development version is tested on Linux operating systems. (CentOS 7 and Ubuntu 16.04). 

