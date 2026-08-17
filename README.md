# CALVADOS dsRNA and dsDNA

This repository contains Python code, [Jupyter](http://jupyter.org) Notebooks, and data for reproducing the results presented in the manuscript "Coarse-grained models for simulations of double-stranded nucleic acids for mixed protein-nucleic acid condensates".

### Layout
- `figure`: a Notebook to reproduce all figures and the data required to run it.
- `code`: code used to (1) build double-stranded nucleic acids, (2) run simulations using the CALVADOS package, and (3) analyze trajectories.  

### Examples
- `code/md/ddx4_dsrna/flexible/slab_DDX4-400_dsACUG-15_direct_l013_k10` contains simulation files for a Ddx4N1 and 24-bp RNA system. Calibrated values are provided for the base lambda, elastic network force constant, and cutoff distance.
- `code/md/ddx4_dsdna/slab_DDX4_dsACTG24_l065/` contains simulation files for a Ddx4N1 and 24-bp DNA system. Calibrated values are provided for the base lambda, elastic network force constant, and cutoff distance.


To open the Notebook, install [Miniconda](https://conda.io/miniconda.html) and make sure all required packages are installed by issuing the following terminal commands
```bash
    conda env create -f environment.yml
    source activate dsrna
    jupyter-notebook
```

To run simulations, install [the CALVADOS package](https://github.com/KULL-Centre/CALVADOS.git). 
The `examples` folder in the package contains dsRNA simulations. 
DNA simulations can be run using the "rna" component.
  
