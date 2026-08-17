import mdtraj as md
import numpy as np
import pandas as pd
from numba import jit
import string
from scipy.ndimage import gaussian_filter1d
from mdtraj import element
import os
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from cycler import cycler
#from matplotlib.colors import LogNorm
import warnings
import itertools
warnings.filterwarnings('ignore')
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from statsmodels.tsa.stattools import acf
import sys
#! git clone https://github.com/fpesceKU/BLOCKING.git
sys.path.append('BLOCKING')
from main import BlockAnalysis
from openmm.app import PDBFile


ddx4_only = False
ddx4_24bp_ssRNA_small = False
ddx4_24bp_ssRNA_large =False
ddx4_24bp_ssRNA_large_ssrna_8 = False
ddx4_24bp_dsRNA_small = False
ddx4_24bp_dsRNA_large = True
ddx4_24bp_dsRNA_large_flexible = False

CAPRIN1_WT_ssRNA_dsRNA = False
CAPRIN_WT_ssRNA_label =  False
CAPRIN1_RK_ssRNA_dsRNA = False
CAPRIN_RK_ssRNA_label =  False
CAPRIN1_WT_ssRNA_dsRNA_80mM = False
CAPRIN_WT_ssRNA_label_80mM =  False
CAPRIN1_WT_ssRNA_dsRNA_100mM = False
CAPRIN_WT_ssRNA_label_100mM =  False

ddx4_24bp_ssDNA_large = False
ddx4_24bp_dsDNA = False
"""
Protein 1, Protein 2 and anothoer component system
this code calculate histgram for each component
"""

RNA_LIST =  ['ss_acug','acug',
             'ss_acug24','acug24',
             'sspolyR12','dspolyR12']

def hist_comp(name, atom_indices, n_seq):
    """
    this function loads dcd and return histogram in mM
    """
    print("Atom indicies",atom_indices)
    trj = md.load(f'traj/{name}.dcd', top=f'traj/{name}.pdb', atom_indices=atom_indices)
    z0 = trj.unitcell_lengths[0,2]
    zs = np.arange(0,z0,1) # bin size 1 nm
    pos = trj.xyz
    print("Traj shape",pos.shape) 
    ns = []
    for t in range(trj.n_frames):
        n,_ = np.histogram(pos[t,:,2],bins=zs)
        ns.append(n)
    ns = np.array(ns)
    L = trj.unitcell_lengths[0,0]
    N = n_seq
    dLz = zs[1]-zs[0] 
    conv = 10/6.022/N/L/L/dLz*1e3 # in mM #conv = 100/6.022/N/L/L*1e3 # in mM
    xs = 0.5*(zs[1:]+zs[:-1]) 
    return xs, ns*conv


def hist_comp_sphere(name, atom_indices, n_seq):
    """
    using mdtraj
    trj = md.load(f'traj/{name}.dcd', top=f'traj/{name}.pdb', atom_indices=atom_indices)    
    r0 = trj.unitcell_lengths[0,2]
    rs = np.arange(0,r0/2,1)
    pos = trj.xyz
    center = np.array([r0/2,r0/2,r0/2])
    pos = np.linalg.norm(pos-center,axis=-1)
    print("Traj shape",pos.shape)
    ns = []
    for t in range(trj.n_frames):
        n,_ = np.histogram(pos[t,:],bins=rs)
        ns.append(n)
    ns = np.array(ns)
    N = n_seq
    xs = 0.5*(rs[1:]+rs[:-1])
    A = 4*np.pi*np.square(xs)
    dr = rs[1]-rs[0]
    conv = 10/6.022/N/A/dr*1e3 # in mM 
    return xs, ns*conv
    """
    #u = mda.Universe(f"traj/{name}.pdb", f"traj/{name}.dcd")
    pdb = PDBFile(f"traj/{name}.pdb")
    u = mda.Universe(pdb.topology,f"traj/{name}.dcd",in_memory=True,topology_format="OPENMMTOPOLOGY")

    
    u.trajectory[0]
    r0 = u.trajectory[0].dimensions[2] / 10  # unit conversion 1A -> nm
    rs = np.arange(0, r0 / 2, 1) # bin size 1nm
    center = np.array([r0/2, r0/2, r0/2])
    
    pos_example = u.atoms[atom_indices].positions / 10  # unit conversion 1A -> nm
    print("Example atom positions shape:", pos_example.shape)
    print("Total number of frames:", len(u.trajectory))
    
    ns = []
    for ts in u.trajectory:
        pos = u.atoms[atom_indices].positions / 10  # unit conversion 1A -> nm
        distances = np.linalg.norm(pos - center, axis=1)
        n, _ = np.histogram(distances, bins=rs)
        ns.append(n)
    
    ns = np.array(ns)
    xs = 0.5 * (rs[1:] + rs[:-1])
    A = 4 * np.pi * np.square(xs)
    dr = rs[1] - rs[0]
    conv = 10 / 6.022 / n_seq / A / dr * 1e3  # unit in mM
    
    return xs, ns * conv


def get_droplet_pos(h,rc1=None,rc2=None):
    """
    this function return bools to indicate dense/dilute phase
    """
    lz = h.shape[1]+1
    edges = np.arange(-lz/2.,lz/2.,1)/1 # bin 1 nm
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d)
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h,axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    p0=[1,1,1,1]

    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4))
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))

    cutoffs1 = [res1.x[2]-rc1*res1.x[3],-res2.x[2]+rc1*res2.x[3]]
    cutoffs2 = [res1.x[2]+rc2*res1.x[3],-res2.x[2]-rc2*res2.x[3]]

    bool1 = np.logical_and(z<cutoffs1[0],z>cutoffs1[1])
    bool2 = np.logical_or(z>cutoffs2[0],z<cutoffs2[1])

    return bool1, bool2, np.array([cutoffs1[0],cutoffs1[1],cutoffs2[0],cutoffs2[1]])


def get_droplet_pos_cubic(h,rc1=None,rc2=None):
    """
    this function return bools to indicate dense/dilute phase
    """
    lz = h.shape[1]+1
    edges = np.arange(0,lz,1)/1 # bin 1 nm
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d)
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h,axis=0)
    p0=[1,1,1,1]

    res1 = least_squares(residuals, x0=p0, args=[z, hm], bounds=([0]*4,[100]*4))

    cutoffs1 = [res1.x[2]-rc1*res1.x[3]] # dense
    cutoffs2 = [res1.x[2]+rc2*res1.x[3]] # dilute

    bool1 = z<cutoffs1[0]
    bool2 = z>cutoffs2[0]

    return bool1, bool2, np.array([cutoffs1[0],cutoffs2[0]])


def get_conc_by_pos(h,bool1,bool2):
    """
    this function uses bool1 and bool2 to calculate conc
    """
    dilarray = np.apply_along_axis(lambda a: a[bool2].mean(), 1, h)
    denarray = np.apply_along_axis(lambda a: a[bool1].mean(), 1, h)

    hm = np.mean(h,axis=0)
    dil = hm[bool2].mean()
    den = hm[bool1].mean()

    #print(dilarray)
    block_dil = BlockAnalysis(dilarray)
    block_den = BlockAnalysis(denarray)
    block_dil.SEM()
    block_den.SEM()
    return np.mean(dilarray), block_dil.sem, np.mean(denarray), block_den.sem


def compute_hist_conc(name,name_list,index_dic,seq_len_dic,ref_name,droplet=False,rc1=4,rc2=8):
    """
    rc1=3,rc2=6  2025/08/12
    rc1=4,rc2=8  2025/08/12
    """
    if not os.path.exists('data'):
        os.system('mkdir data')
    # compute hist
    for comp_name in name_list:
        comp_index = index_dic[comp_name]
        comp_seq = seq_len_dic[comp_name]
 
        if comp_name in RNA_LIST:
            n_bead = comp_seq*2
        else:
            n_bead = comp_seq
        if droplet:
            print("droplet detected ===========")
            x,y = hist_comp_sphere(name,comp_index,n_bead)
        else:
            print("slab detected ===========")
            x,y = hist_comp(name,comp_index,n_bead)

        np.save(f"data/hist_{name}_{comp_name}.npy", y)

    # comput conc
    y_ref = np.load(f"data/hist_{name}_{ref_name}.npy")
    
    if droplet:
        x_den_bool,x_dil_bool,cutoff_pos = get_droplet_pos_cubic(y_ref,rc1=0.25,rc2=4)
    else:
        x_den_bool,x_dil_bool,cutoff_pos = get_droplet_pos(y_ref,rc1=rc1,rc2=rc2) # 20250725
        #x_den_bool,x_dil_bool,cutoff_pos = get_droplet_pos(y_ref,rc1=4,rc2=8) # 20250725
    np.save(f"data/cutoff_{name}.npy",cutoff_pos)
    print("phase boundary pos",cutoff_pos)
    for comp_name in name_list:
        y = np.load(f"data/hist_{name}_{comp_name}.npy")   
        csat_g, csat_err_g, cden_g, cden_err_g = get_conc_by_pos(y, x_den_bool, x_dil_bool)
        np.save(f"data/conc_{name}_{comp_name}.npy",np.array([csat_g, csat_err_g, cden_g, cden_err_g]))    


def read_fasta_to_dict(file_path):
    protein_dict = {}
    with open(file_path, 'r') as file:
        protein_name = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                protein_name = line[1:]  # Remove the ">" and take the name
                protein_dict[protein_name] = ""  # Initialize the sequence as an empty string
            else:
                protein_dict[protein_name] += line  # Append the sequence to the respective protein
    return protein_dict



def get_seq_length(name):
    fn = './fasta.fasta'
    protein_dict = read_fasta_to_dict(fn)
    seq = protein_dict[name]
    return len(seq)



def get_index(comp_list,comp_n_dic,seq_len_dic,comp_list_sort=True): 
    if comp_list_sort:
        comp_list.sort()    

    # make index list
    index_dic = {}
    start = 0
    for comp_name in comp_list:
        seq_len = seq_len_dic[comp_name]
        comp_n = comp_n_dic[comp_name]
        if comp_name in RNA_LIST:
            index = np.arange(start,start+2*comp_n*seq_len) 
        else:
            index = np.arange(start,start+comp_n*seq_len)  
        start = index[-1]+1
        index_dic[comp_name] = index
    return index_dic 


def main(comp_list,comp_n_dic,ref_name,comp_list_sort=True):

    # generate seq_len_list
    seq_len_dic = {}
    for comp_name in comp_list:
        seq_len_dic[comp_name] = get_seq_length(comp_name)
        print("seq_len_dic",seq_len_dic) 
    # generate index list
    index_dic = get_index(comp_list,comp_n_dic,seq_len_dic,comp_list_sort=comp_list_sort)
    for comp_name in comp_list:
        print(comp_name,"index",index_dic[comp_name][0]," to ",index_dic[comp_name][-1],"seq len",seq_len_dic[comp_name])

    # compute histogram and concentration
    name=''
    for i,comp_name in enumerate(comp_list):
        if i==0:
            name +=  f'{comp_name}-{comp_n_dic[comp_name]}'
        else:
            name +=  f'_{comp_name}-{comp_n_dic[comp_name]}'
    print(name)

    return name,comp_list,index_dic,seq_len_dic,ref_name
    #compute_hist_conc(name,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet)    
    #print("="*50)



#=============================================================================================--
#=============================================================================================--
#=============================================================================================--

droplet =False    # this is False for slab simulations 

# DDX4-only slab simulation
#ddx4_only = True
comp1_name = "DDX4"
comp1_n = 350

if ddx4_only:
    name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name], {comp1_name:comp1_n},comp1_name)
    compute_hist_conc(name,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=6 )   
    print("="*100)


# small DDX4-RNA systems (400 chains)
# 24 bps RNA
#ddx4_24bp_ssRNA_small = True
comp1_name="DDX4"
comp2_name="ss_acug24"
comp1_n = 400

if ddx4_24bp_ssRNA_small:
    for comp2_n in [8,16,24,32]: 
        name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)
        compute_hist_conc(name,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=5,rc2=6 )   
        print("="*100)



# large DDX4-RNA systems (800 chains)
# 24 bps RNA
#ddx4_24bp_ssRNA_large = True
comp1_name = 'DDX4'
comp2_name = 'ss_acug24'
comp1_n = 800
comp2_n = 16

name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)

if ddx4_24bp_ssRNA_large:
    for key in ["_1","_2","_3","_4","_5"]: 
        print("name+key:", name+key)
        compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
        print("="*100)

# large DDX4-RNA systems (800 chains)
# 24 bps RNA
#ddx4_24bp_ssRNA_large = True
comp1_name = 'DDX4'
comp2_name = 'ss_acug24'
comp1_n = 800
comp2_n = 8

name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)

if ddx4_24bp_ssRNA_large_ssrna_8:
    for key in ["_1","_2","_3","_4"]:
        print("name+key:", name+key)
        compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
        print("="*100)


# small DDX4-dsRNA systems (300 chains)
# 24 bps RNA in doudle strand 
#ddx4_24bp_dsRNA_small = True
comp1_name = "DDX4"
comp2_name = "ds_acug24" # acug24 is dsRNA
comp1_n = 300
comp2_n = 15

name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n}, comp1_name)

if ddx4_24bp_dsRNA_small:
    for l in ['00','01','02','03']:
        compute_hist_conc(name+f'_direct_l{l}',comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=6)
        print("="*100)



# large DDX4-dsRNA systems (400 chains)
# 24 bps RNA in doudle strand 
#ddx4_24bp_dsRNA_large = True
comp1_name="DDX4"
comp2_name="ds_acug24"
comp1_n = 400 
comp2_n = 15
name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)

if ddx4_24bp_dsRNA_large:
    #for l in ['00','01','013','02','03']:
    for l in ['02']:
        compute_hist_conc(name+f'_direct_l{l}',comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=6)
        print("="*100)



# large DDX4-dsRNA systems (400 chains)
# 24 bps RNA in doudle strand "flexible"
#ddx4_24bp_dsRNA_large_flexible = True
comp1_name = 'DDX4'
comp2_name = 'ds_acug24'
comp1_n = 400
comp2_n = 15

name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)

if ddx4_24bp_dsRNA_large_flexible:
    for l in ['l013_k10','l01_k10','l02_k10']:
        name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)
        compute_hist_conc(name+f'_direct_{l}',comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=6)
        print(f'_direct_{l}')



# CAPRIN1 systems
# CAPRIN1_WT + ssRNA + dsRNA
#CAPRIN1_WT_ssRNA_dsRNA = True
comp1_name = 'CAPRIN1_N623TN630T'
comp2_name = 'sspolyR12'
comp3_name = 'dspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 15

if CAPRIN1_WT_ssRNA_dsRNA:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_200mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=8)
    print("="*100)

if CAPRIN1_WT_ssRNA_dsRNA_80mM:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_80mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=8)
    print("="*100)

if CAPRIN1_WT_ssRNA_dsRNA_100mM:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_100mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=8)
    print("="*100)


# CAPRIN1_WT + ssRNA + labeled_ssRNA
#CAPRIN_WT_ssRNA_label = True
comp1_name = 'CAPRIN1_N623TN630T'
comp2_name = 'sspolyR12'
comp3_name = 'label_sspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 30

if CAPRIN_WT_ssRNA_label:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_200mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=8)
    print("="*100)

if CAPRIN_WT_ssRNA_label_80mM:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_80mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=8)
    print("="*100)

if CAPRIN_WT_ssRNA_label_100mM:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_100mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=8)
    print("="*100)


# CAPRIN1_RK + ssRNA + dsRNA
#CAPRIN1_RK_ssRNA_dsRNA = True
comp1_name = 'CAPRIN1_N623TN630T_RK'
comp2_name = 'sspolyR12'
comp3_name = 'dspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 15

if CAPRIN1_RK_ssRNA_dsRNA:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_80mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=4)
    print("="*100)




# CAPRIN1_RK + ssRNA + labeled_ssRNA
#CAPRIN_RK_ssRNA_label = True
comp1_name = 'CAPRIN1_N623TN630T_RK'
comp2_name = 'sspolyR12'
comp3_name = 'label_sspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 30

if CAPRIN_RK_ssRNA_label:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic,ref_name = main(comp_list, comp_n_dic, comp1_name,comp_list_sort=False)

    key="_80mM"
    compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=2,rc2=4)
    print("="*100)


# large DDX4-DNA systems (800 chains)
# 24 bps DNA
comp1_name = 'DDX4'
comp2_name = 'ssACTG24'
comp1_n = 800
comp2_n = 8

name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)

if ddx4_24bp_ssDNA_large:
    for key in ["_1","_2","_3","_4"]:
        print("name+key:", name+key)
        compute_hist_conc(name+key,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
        print("="*100)


# DDX4-dsDNA systems (400 chains)
# 24 bps DNA
comp1_name = 'DDX4'
comp2_name = 'dsACTG24'
comp1_n = 400
comp2_n = 15

name,comp_list,index_dic,seq_len_dic,ref_name = main( [comp1_name,comp2_name], {comp1_name:comp1_n,comp2_name:comp2_n},comp1_name)

if ddx4_24bp_dsDNA:
    print("name:", name)
    compute_hist_conc(name,comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l02",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l03",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l04",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l07",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l06",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l08",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    compute_hist_conc(name+"_l065",comp_list,index_dic,seq_len_dic,ref_name,droplet=droplet,rc1=3,rc2=5)
    print("="*100)
