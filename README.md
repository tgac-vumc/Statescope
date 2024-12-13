<p align="center">
  <img width="424" height="200" src="https://github.com/tgac-vumc/Statescope/blob/main/img/Logo_Statescope.png">
</p>

# OncoBLADE: Malignant cell fraction-informed deconvolution
OncoBLADE is a Bayesian deconvolution method designed to estimate cell type-specific gene expression profiles and fractions from bulk RNA profiles of tumor specimens by integrating prior knowledge on cell fractions. You can find the [preprint of OncoBLADE at Research Square](https://www.researchsquare.com/article/rs-4252952/v1).

<p align="center">
  <img width="75%" height="75%" src="https://github.com/tgac-vumc/Statescope/blob/main/img/Statescope_Overview.png">
</p>


## Installation

### Using pip

The python package of BLADE is available on pip.
You can simply (takes only <1min):

```
pip install OncoBLADE
```

We tested BLADE with `python => 3.6`.


### Using Conda

One can create a conda environment contains BLADE and also other dependencies to run [Demo](https://github.com/tgac-vumc/BLADE/blob/master/jupyter/BLADE%20-%20Demo%20script.ipynb).
The environment definition is in [environment.yml](https://github.com/tgac-vumc/BLADE/environment.yml).

### Step 1: Installing Miniconda 3
First, please open a terminal or make sure you are logged into your Linux VM. Assuming that you have a 64-bit system, on Linux, download and install Miniconda 3 with:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
On MacOS X, download and install with:

```
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### Step 2: Create a conda environment

You can install all the necessary dependency using the following command (may takes few minutes; `mamba` is quicker in general).

```
conda env create --file environment.yml
```

Then, the `OncoBLADE` environment can be activate by:

```
conda activate OncoBLADE
```

### Step 3: Running a demo script

You can find a demo script under `jupyter` folder.
You can open the script using the command below after activating the `OncoBLADE` environment:

```
jupyter notebook jupyter/OncoBLADE\ -\ Demo script.ipynb
```


#### Demo notebook is available under `jupyter`. See below how to open it. 


## System Requirements

### Hardware Requirements

OncoBLADE can run on the minimal computer spec, such as Binder (1 CPU, 2GB RAM on Google Cloud), when data size is small. However, OncoBLADE can significantly benefit from the larger amount of CPUs and RAM.

### OS Requirements

The package development version is tested on Linux operating systems. (CentOS 7 and Ubuntu 16.04). 

