import numpy as np
import mdtraj as md
import mdna

seq='ACTGACTGACTGACTGACTGACTG'
dna = mdna.make(sequence=seq, n_bp=24)

dna.describe()
dna.save_pdb('./actg_bp24')
