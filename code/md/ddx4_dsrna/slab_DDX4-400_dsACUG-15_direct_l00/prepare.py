import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

#parser = ArgumentParser()
#parser.add_argument('--name',nargs='?',required=True,type=str)
#args = parser.parse_args()

cwd = os.getcwd()
sysname = 'md'

## set the side length of the cubic box
L =  30
Lz = 210
# set the saving interval (number of integration steps)
N_save = 1000000

# set final number of frames to save
N_frames = 1600

residues_file = f'{cwd}/input/residues_C2RNA.csv'
fasta_file = f'{cwd}/input/fastalib.fasta'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, Lz], # nm
  temp = 293, # K
  ionic = 0.15, # molar
  pH = 8.0,
  topol = 'random',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA', # or CUDA
  gpu_id = 1,
  threads = 1,
  restart = 'checkpoint',
  frestart = 'restart.chk',
  slab_eq = True,
  k_eq = 0.01,
  steps_eq = 1000000,
  verbose = True,
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)
#subprocess.run(f'mkdir -p data',shell=True)

#analyses = f"""
#
#from calvados.analysis import save_conf_prop
#
#save_conf_prop(path="{path:s}",name="{sysname:s}",residues_file="{residues_file:s}",output_path=f"{cwd}/data",start=100,is_idr=False,select='all')
#"""

config.write(path,name='config.yaml')

components = Components(
  ffasta = fasta_file,
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = True, # apply restraints
  charge_termini = 'both', # charge N or C or both
  # INPUT
  fresidues = residues_file, # residue definitions
  fdomains = f'{cwd}/input/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files
  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2)
  k_harmonic = 7000., # Restraint force constant

  # RNA settings
  rna_kb1 = 1400.0,
  rna_kb2 = 2200.0,
  rna_ka = 4.20,
  rna_pa = 3.14,
  rna_nb_sigma = 0.4,
  rna_nb_scale = 15,
  rna_nb_cutoff = 2.0

)

components.add(name='DDX4',molecule_type='protein', nmol=400,restraint=False,ext_restraint=True)

components.add(name='acug',molecule_type='rna', nmol=15, restraint=True,
               restraint_type='harmonic', k_harmonic=7000, cutoff_restr=1.5,
               use_com = False)

components.write(path,name='components.yaml')


