import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations
from openmm.app import PDBFile


start = 100 # 100 frames * 10 ns_frame = 1 us
ddx4_only = False
ddx4_24bp_ssRNA_small = False #True
ddx4_24bp_ssRNA_large = False #True
ddx4_24bp_ssRNA_large_ssrna_8 = False
ddx4_24bp_dsRNA_small = False #True
ddx4_24bp_dsRNA_large = True
ddx4_24bp_dsRNA_large_flexible = False #True

CAPRIN1_WT_ssRNA_dsRNA = False
CAPRIN_WT_ssRNA_label =  False
CAPRIN1_WT_ssRNA_dsRNA_80mM = False
CAPRIN_WT_ssRNA_label_80mM =  False
CAPRIN1_WT_ssRNA_dsRNA_100mM = False
CAPRIN_WT_ssRNA_label_100mM =  False

CAPRIN1_RK_ssRNA_dsRNA = False
CAPRIN_RK_ssRNA_label =  False

ddx4_24bp_ssDNA_large_ssrna_8 = False
ddx4_24bp_dsDNA = False

RNA_LIST =  ['ss_acug','acug',
             'ss_acug24','acug24',
             'sspolyR12','dspolyR12']

def calc_zpatch(z,h,cutoff):
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = []
    hpatch = []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch


def center_slab(path,name,outname,start=None,end=None,step=1,input_pdb='top.pdb',ref_atoms_index=None,h_cutoff=None):
    print(path)
    if not os.path.exists('traj'):
        os.system('mkdir traj')

    u = MDAnalysis.Universe(path+f'/{input_pdb}',path+f'/{name}.dcd',in_memory=True)
    os.system(f'cp {path}/{input_pdb} traj/{outname}.pdb')
    n_frames = len(u.trajectory[start:end:step])
    ag = u.atoms
    n_atoms = ag.n_atoms
    print(outname, "Total", 10**-3*(u.trajectory[-1].time),"ns,", n_frames, 'frames')
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,10) # bin size 10 angstrom
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    with MDAnalysis.Writer(f'traj/{outname}.dcd',n_atoms) as W:
        for t,ts in enumerate(u.trajectory[start:end:step]):
            # shift max density to center
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos[ref_atoms_index],bins=edges)
            if t==0:
                print(f"histogram bin size {str(0.1*(z[1]-z[0]))[:5]} nm\n",h)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos[ref_atoms_index], bins=edges)
            zpatch, hpatch = calc_zpatch(z,h,h_cutoff)
            zmid = np.average(zpatch,weights=hpatch)
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            W.write(ag)


def center_cubic(path,name,outname,start=None,end=None,step=1,input_pdb='top.pdb',ref_atoms_index=None,h_cutoff=None,res_1000k=False):
    print(path)
    if not os.path.exists('traj'):
        os.system('mkdir traj')

    os.system(f'cp {path}/{input_pdb} traj/{outname}.pdb')
    if res_1000k: # use openmm reader when resid > 100,000
        pdb = PDBFile(path+f'/{input_pdb}')
        u = MDAnalysis.Universe(pdb.topology,path+f'/{name}.dcd',in_memory=True,topology_format="OPENMMTOPOLOGY")  
        print(">100k residues deteted !!", u)
    else:  
        os.system(f'cp {path}/{input_pdb} traj/{outname}.pdb')
        u = MDAnalysis.Universe(path+f'/{input_pdb}',path+f'/{name}.dcd',in_memory=True)

    n_frames = len(u.trajectory[start:end:step])
    ag = u.atoms
    n_atoms = ag.n_atoms
    print(outname, "Total", 10**-3*(u.trajectory[-1].time),"ns,", n_frames, 'frames')
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,10) # bin size 10 angstrom
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    with MDAnalysis.Writer(f'traj/{outname}.dcd',n_atoms) as W:  
        for t,ts in enumerate(u.trajectory[start:end:step]):
            ag.translate([lz,lz,lz])
            ts = transformations.wrap(ag)(ts)

            d_pos = []
            for axis in [0,1,2]:
                zpos = ag.positions.T[axis]
                h, e = np.histogram(zpos[ref_atoms_index],bins=edges)
                if t==0:
                    print(f"histogram bin size {str(0.1*(z[1]-z[0]))[:5]} nm\n",h)
                zmax = z[np.argmax(h)]
                d_pos.append(-zmax+0.5*lz)
            ag.translate(np.array(d_pos))
            ts = transformations.wrap(ag)(ts)
   
            d_pos = [] 
            for axis in [0,1,2]:
                zpos = ag.positions.T[axis]
                h, e = np.histogram(zpos[ref_atoms_index],bins=edges)
                if t==0:
                    print(f"histogram bin size {str(0.1*(z[1]-z[0]))[:5]} nm\n",h)
                zpatch, hpatch = calc_zpatch(z,h,h_cutoff)
                zmid = np.average(zpatch,weights=hpatch)
                d_pos.append(-zmid+0.5*lz)
            ag.translate(np.array(d_pos))
            ts = transformations.wrap(ag)(ts)
            W.write(ag)



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


def get_seq_length(name):
    fn = './fasta.fasta'
    protein_dict = read_fasta_to_dict(fn)
    seq = protein_dict[name]
    return len(seq)


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


def prepare_inputs(comp_list,comp_n_dic,comp_list_sort=True):
    seq_len_dic = {}
    for comp_name in comp_list:
        seq_len_dic[comp_name] = get_seq_length(comp_name)

    index_dic  = get_index(comp_list,comp_n_dic,seq_len_dic,comp_list_sort)

    name = ''
    for i,comp_name in enumerate(comp_list):
        if i==0:
            name += f'{comp_name}-{comp_n_dic[comp_name]}'
        else:
            name += f'_{comp_name}-{comp_n_dic[comp_name]}'

    return seq_len_dic, index_dic, name


#**************************************************
#**************************************************
#**************************************************
#**************************************************
#**************************************************
#**************************************************
#**************************************************
#**************************************************
#**************************************************


# DDX4-only slab simulation
#ddx4_only = False            
comp1_name = 'DDX4'
comp1_n = 350
comp_list = [comp1_name]
comp_n_dic = {comp1_name:comp1_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_only:
    center_slab('../md/DDX4-350/DDX4/',
                'DDX4',
                name,
                start=start, end=800, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)
    print("="*100)


# small DDX4-RNA systems (400 chains)
# 24 bps RNA
#ddx4_24bp_ssRNA_small = False
comp1_name = 'DDX4'
comp2_name = 'ss_acug24'
comp1_n = 400

if ddx4_24bp_ssRNA_small:
    for comp2_n in [8,16,24,32]: 
        comp_list = [comp1_name,comp2_name]
        comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
        seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)
        center_slab(f'../md/slab_DDX4_ssACUG24_grid/slab_DDX4-{comp1_n}_ssACUG24-{comp2_n}/md/',
                'md',
                name,
                start=start, end=1200, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)
        print("="*100)


# large DDX4-RNA systems (800 chains)
# 24 bps RNA
#ddx4_24bp_ssRNA_large = False
comp1_name = 'DDX4'
comp2_name = 'ss_acug24'
comp1_n = 800
comp2_n = 16

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_ssRNA_large:
    for key in ["_1","_2","_3","_4","_5"]:
        center_slab(f'../md/slab_DDX4_ssACUG24_grid/slab_DDX4-{comp1_n}_ssACUG24-{comp2_n}{key}/md/',
                    'md',
                    name+key,
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)

        print("="*100)


# large DDX4-RNA systems (800 chains)
# 24 bps RNA
#ddx4_24bp_ssRNA_large = False
comp1_name = 'DDX4'
comp2_name = 'ss_acug24'
comp1_n = 800
comp2_n = 8

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_ssRNA_large_ssrna_8:
    for key in ["_1","_2","_3","_4"]:
        center_slab(f'../md/slab_DDX4_ssACUG24_grid/slab_DDX4-{comp1_n}_ssACUG24-{comp2_n}{key}/md/',
                    'md',
                    name+key,
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)

        print("="*100)


# small DDX4-dsRNA systems (300 chains)
# 24 bps RNA in doudle strand 
#ddx4_24bp_dsRNA_small = False
comp1_name = 'DDX4'
comp2_name = 'ds_acug24'
comp1_n = 300
comp2_n = 15

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_dsRNA_small:
    for l in ['00','01','02','03']:
        #center_slab(f'../md/slab_DDX4_dsACUG24_grid/slab_DDX4-{comp1_n}_dsACUG-{comp2_n}_direct_l{l}/md/',
        center_slab(f'../md/slab_DDX4_dsACUG24_grid_update/slab_DDX4-{comp1_n}_dsACUG-{comp2_n}_direct_l{l}/md/',
                    'md',
                    name+f'_direct_l{l}',
                    start=start, end=1600, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)
        print("="*100)


# large DDX4-dsRNA systems (400 chains)
# 24 bps RNA in doudle strand 
#ddx4_24bp_dsRNA_large = False
comp1_name = 'DDX4'
comp2_name = 'ds_acug24'
comp1_n = 400 
comp2_n = 15

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_dsRNA_large:
    #for l in ['00','01','013','02','03']: 
    for l in ['02']: 
        center_slab(f'../md/slab_DDX4_dsACUG24_grid_update/slab_DDX4-{comp1_n}_dsACUG-{comp2_n}_direct_l{l}/md/',
                    'md',
                    name+f'_direct_l{l}',
                    start=start, end=1600, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)
        print("="*100)
#"""


# large DDX4-dsRNA systems (400 chains)
# 24 bps RNA in doudle strand "flexible"
#ddx4_24bp_dsRNA_large_flexible = False
comp1_name = 'DDX4'
comp2_name = 'ds_acug24'
comp1_n = 400
comp2_n = 15

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_dsRNA_large_flexible:
    for key in ['l013_k10','l01_k10','l02_k10']:
        #center_slab(f'../md/slab_DDX4_dsACUG24_k10/slab_DDX4-{comp1_n}_dsACUG-{comp2_n}_direct_{key}/md/',
        center_slab(f'../md/slab_DDX4_dsACUG24_k10_update/slab_DDX4-{comp1_n}_dsACUG-{comp2_n}_direct_{key}/md/',
                        'md',
                        name+f'_direct_{key}',
                        start=start, end=1600, step=1,
                        input_pdb='top.pdb',
                        ref_atoms_index=index_dic[comp1_name],
                        h_cutoff=200)
        print("="*100)

"""
# use backup trajectory
comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)
for key in ['l013_k10','l01_k10']:
    center_slab(f'../md/slab_DDX4_dsACUG24_k10_update/slab_DDX4-{comp1_n}_dsACUG-{comp2_n}_direct_{key}_1/md/',
        'md',
        name+f'_direct_{key}',
        start=start, end=1600, step=1,
        input_pdb='top.pdb',
        ref_atoms_index=index_dic[comp1_name],
        h_cutoff=200)
print("="*100)
"""

"""
# polyR12
start = 100 # 10 ns per frame
comp1_name = 'DDX4'
comp2_name = 'ss_acug12'
comp1_n = 800

for key in ["_1","_2","_3","_4","_5"]:
    comp2_n = 16
    comp_list = [comp1_name,comp2_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)
    center_slab(f'../md/slab_DDX4_ssACUG24_grid/slab_DDX4-{comp1_n}_ssACUG24-{comp2_n}{key}/md/',
            'md',
            name+key,
            start=start, end=1200, step=1,
            input_pdb='top.pdb',
            ref_atoms_index=index_dic[comp1_name],
            h_cutoff=200)
print("="*100)
"""



# CAPRIN1 systems
# CAPRIN1_WT + ssRNA + dsRNA
#CAPRIN1_WT_ssRNA_dsRNA = False
comp1_name = 'CAPRIN1_N623TN630T'
comp2_name = 'sspolyR12'
comp3_name = 'dspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 15

if CAPRIN1_WT_ssRNA_dsRNA:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
    center_slab(f'../caprin_md/CAPRIN1-{comp1_n}_sspolyR12-{comp2_n}_dspolyR12-{comp3_n}_200mM_k10/md/',
                'md',
                name+"_200mM",
                start=start, end=1200, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

if CAPRIN1_WT_ssRNA_dsRNA_80mM:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
    center_slab(f'../caprin_md/CAPRIN1-{comp1_n}_sspolyR12-{comp2_n}_dspolyR12-{comp3_n}_80mM_k10/md/',
                'md',
                name+"_80mM",
                start=start, end=1200, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

if CAPRIN1_WT_ssRNA_dsRNA_100mM:
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
    center_slab(f'../caprin_md/CAPRIN1-{comp1_n}_sspolyR12-{comp2_n}_dspolyR12-{comp3_n}_100mM_k10/md/',
                'md',
                name+"_100mM",
                start=start, end=1200, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)


# CAPRIN1_WT + ssRNA + labeled_ssRNA
#CAPRIN_WT_ssRNA_label = False
comp1_name = 'CAPRIN1_N623TN630T'
comp2_name = 'sspolyR12'
comp3_name = 'label_sspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 30

if CAPRIN_WT_ssRNA_label :
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
    center_slab(f'../caprin_md/CAPRIN1-{comp1_n}_sspolyR12-{comp2_n}_label_sspolyR12-{comp3_n}_200mM_k10/md/',
                    'md',
                    name+"_200mM",
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)

if CAPRIN_WT_ssRNA_label_80mM :
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
    center_slab(f'../caprin_md/CAPRIN1-{comp1_n}_sspolyR12-{comp2_n}_label_sspolyR12-{comp3_n}_80mM_k10/md/',
                    'md',
                    name+"_80mM",
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)



if CAPRIN_WT_ssRNA_label_100mM :
    comp_list = [comp1_name,comp2_name,comp3_name]
    comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
    seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
    center_slab(f'../caprin_md/CAPRIN1-{comp1_n}_sspolyR12-{comp2_n}_label_sspolyR12-{comp3_n}_100mM_k10/md/',
                    'md',
                    name+"_100mM",
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)

# CAPRIN1_RK + ssRNA + dsRNA
#CAPRIN1_RK_ssRNA_dsRNA = True
comp1_name = 'CAPRIN1_N623TN630T_RK'
comp2_name = 'sspolyR12'
comp3_name = 'dspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 15

if CAPRIN1_RK_ssRNA_dsRNA:
    for key in ["_80mM"]:
        comp_list = [comp1_name,comp2_name,comp3_name]
        comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
        seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
        center_slab(f'../caprin_md/CAPRIN1_RK-{comp1_n}_sspolyR12-{comp2_n}_dspolyR12-{comp3_n}{key}_k10/md/',
                    'md',
                    name+key,
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)


# CAPRIN1_RK + ssRNA + labeled_ssRNA
#CAPRIN_RK_ssRNA_label = True
comp1_name = 'CAPRIN1_N623TN630T_RK'
comp2_name = 'sspolyR12'
comp3_name = 'label_sspolyR12'
comp1_n = 500
comp2_n = 500
comp3_n = 30

if CAPRIN_RK_ssRNA_label :
    for key in ["_80mM"]:
        comp_list = [comp1_name,comp2_name,comp3_name]
        comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n,comp3_name:comp3_n}
        seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic,comp_list_sort=False)
        center_slab(f'../caprin_md/CAPRIN1_RK-{comp1_n}_sspolyR12-{comp2_n}_label_sspolyR12-{comp3_n}{key}_k10/md/',
                    'md',
                    name+key,
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)
      


# large DDX4-DNA systems (800 chains)
# 24 bps DNA
comp1_name = 'DDX4'
comp2_name = 'ssACTG24'
comp1_n = 800
comp2_n = 8

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_ssDNA_large_ssrna_8:
    for key in ["_1","_2","_3","_4"]:
        center_slab(f'../dna_md/DDX4-800_ssACTG24-8{key}/md/',
                    'md',
                    name+key,
                    start=start, end=1200, step=1,
                    input_pdb='top.pdb',
                    ref_atoms_index=index_dic[comp1_name],
                    h_cutoff=200)

        print("="*100)


# large DDX4-dsDNA systems (400 chains)
# 24 bp DNA in doudle strand "flexible"
comp1_name = 'DDX4'
comp2_name = 'dsACTG24'
comp1_n = 400
comp2_n = 15

comp_list = [comp1_name,comp2_name]
comp_n_dic = {comp1_name:comp1_n,comp2_name:comp2_n}
seq_len_dic, index_dic, name = prepare_inputs(comp_list,comp_n_dic)

if ddx4_24bp_dsDNA:
    
    """ 
    center_slab(f'../dna_md/slab_DDX4_dsACTG24/md/',
                'md',
                name,
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l02/md/',
                'md',
                name+"_l02",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l03/md/',
                'md',
                name+"_l03",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l04/md/',
                'md',
                name+"_l04",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)
    print("="*100)

    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l06/md/',
                'md',
                name+"_l06",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)
"""

    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l065/md/',
                'md',
                name+"_l065",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

"""
    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l07/md/',
                'md',
                name+"_l07",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)

    center_slab(f'../dna_md/slab_DDX4_dsACTG24_l08/md/',
                'md',
                name+"_l08",
                start=start, end=1600, step=1,
                input_pdb='top.pdb',
                ref_atoms_index=index_dic[comp1_name],
                h_cutoff=200)
    print("="*100)
"""
