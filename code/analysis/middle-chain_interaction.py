import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import warnings
import mdtraj as md
import csv
import os
from multiprocessing import Pool
from mdtraj import element
import pandas as pd

skip = 1 # 100 for test, 10 for production

CAPRIN1_WT_ssRNA_dsRNA = True
CAPRIN1_WT_ssRNA_label_ssRNA = True
CAPRIN1_RK_ssRNA_dsRNA = True
CAPRIN1_RK_ssRNA_label_ssRNA = True
CAPRIN1_WT_ssRNA_dsRNA_80mM = True
CAPRIN1_WT_ssRNA_label_ssRNA_80mM = True
CAPRIN1_WT_ssRNA_dsRNA_100mM = True
CAPRIN1_WT_ssRNA_label_ssRNA_100mM = True

RNA_LIST =  ['ss_acug24','acug24',
             'sspolyR12','dspolyR12','label_sspolyR12']


def read_fasta_to_dict(file_path):
    """
    This funtion returns dictionary of fasta
    """
    protein_dict = {}
    with open(file_path, 'r') as file:
        protein_name = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                protein_name = line[1:]  # Remove the ">" and take the name
            else:
                if protein_name not in RNA_LIST:
                    tmp = line
                else:
                    tmp = ''.join(['p'+s for s in line ]) # RNA sequence
                protein_dict[protein_name] = tmp
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
        #if comp_name in RNA_LIST:
        #    index = np.arange(start,start+2*comp_n*seq_len)
        #else:
        index = np.arange(start,start+comp_n*seq_len)
        start = index[-1]+1
        index_dic[comp_name] = index
    return index_dic


#def main(comp_list,comp_n_dic,ref_name,comp_list_sort=True):
#
#    # generate seq_len_list
#    seq_len_dic = {}
#    for comp_name in comp_list:
#        seq_len_dic[comp_name] = get_seq_length(comp_name)
#        print("seq_len_dic",seq_len_dic)
#    # generate index list
#    index_dic = get_index(comp_list,comp_n_dic,seq_len_dic,comp_list_sort=comp_list_sort)
#    for comp_name in comp_list:
#        print(comp_name,"index",index_dic[comp_name][0]," to ",index_dic[comp_name][-1],"seq len",seq_len_dic[comp_name])

    # compute histogram and concentration
#    name=''
#    for i,comp_name in enumerate(comp_list):
#        if i==0:
#            name +=  f'{comp_name}-{comp_n_dic[comp_name]}'
#        else:
#            name +=  f'_{comp_name}-{comp_n_dic[comp_name]}'
#    print(name)
#
#    return name,comp_list,index_dic,seq_len_dic,ref_name



def make_new_top(fasta_list,df,prot_list):
    '''
    prot_list: [ [prot1_name,prot1_n], [prot2_name,prot2_n], ... ]
    '''
    top = md.Topology()
    for [prot_name,prot_n] in prot_list:
        fasta = fasta_list[prot_name]
        masses = []
        radii = []
        for aa in fasta:
            mass = df.loc[aa, 'MW']
            radius = df.loc[aa, 'sigmas'] / 2
            masses.append(mass)
            radii.append(radius)
        masses[0] += 2
        masses[-1] += 16

        for _ in range(prot_n):
            chain = top.add_chain()
            for i,resname in enumerate(fasta):
                residue = top.add_residue('C{:d}'.format(chain.index), chain, resSeq=chain.index)
                # add an element with unique name to the dictionary. the letter A is prepended to avoid doubles (e.g. cysteine and carbon)
                element.Element._elements_by_symbol.pop('A'+resname.upper(), None)
                el = element.Element.__new__(element.Element, 1, 'A'+resname.upper(), 'A'+resname.upper(), masses[i], radii[i])
                atom = top.add_atom('A'+resname, element=el, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))
    return  top


def read_sigma_lambda(df,fasta1,fasta2):
    # create rc matrix
    print('fasta1:', fasta1)
    print('fasta2:', fasta2)
    sigmas = np.zeros( (len(fasta1)*len(fasta2)) )
    lambdas = np.zeros( (len(fasta1)*len(fasta2)) )
    qqs = np.zeros( (len(fasta1)*len(fasta2)) )
    pairs = [[i, j] for i in list(fasta1) for j in list(fasta2)]
    for i,[s1,s2] in enumerate(pairs):
        sigma1 = df.loc[s1,'sigmas']
        sigma2 = df.loc[s2,'sigmas']
        sigmas[i] = 0.5*(sigma1+sigma2)

        lambda1 = df.loc[s1,'lambdas']
        lambda2 = df.loc[s2,'lambdas']
        lambdas[i] = 0.5*(lambda1+lambda2)

        qqs[i] = df.loc[s1,'q'] * df.loc[s2,'q']
    return sigmas, lambdas, qqs


def extract_middle_chain(df,fasta_list,trj,prot_name,chain_n):
    fasta = fasta_list[prot_name] 
    # make a new topology
    top_new = make_new_top(fasta_list,df,[[prot_name,chain_n]])
    trj =  md.Trajectory(trj.xyz, top_new, trj.time, trj.unitcell_lengths, trj.unitcell_angles) 

    ## placing trajctory to center
    #trj.xyz -= trj.unitcell_lengths[0,:]/2
    trj.make_molecules_whole(inplace=True)

    # compute center of mass
    middle_dist = np.zeros((chain_n,trj.n_frames))
    for j in range(chain_n):
        sel = np.arange(len(fasta)*j,len(fasta)*(j+1)) 
        tmp = trj.atom_slice(sel)
        # for checking purpose
        tmp_z =  tmp.xyz[:,:,2]
        if np.max( np.max(tmp_z,axis=1)-np.min(tmp_z,axis=1) ) > trj.unitcell_lengths[0,2]/2:
            print('check pbc')

        pos_com = md.compute_center_of_mass(tmp)
        middle_dist[j,:] = np.abs( pos_com[:,2] )         

    middle_chain = np.argmin(middle_dist,axis=0) # indices of chains at the center of the slab
    del middle_dist
    
    # make new traj for middle chain 
    pos = []
    for ts in range(trj.n_frames): 
        j = middle_chain[ts]
        sel = np.arange(len(fasta)*j,len(fasta)*(j+1))
        pos.append(trj.atom_slice(sel).xyz[ts])
    
    return np.array(pos), middle_chain


def extract_non_middle_chain(df,fasta_list,trj,prot_name,chain_n,middle_chain_id):
    '''
    for client-client interactions
    output: pos of (chain_n -1) chains
    '''
    fasta = fasta_list[prot_name]
    # make a new topology
    top_new = make_new_top(fasta_list,df,[[prot_name,chain_n]])
    trj =  md.Trajectory(trj.xyz, top_new, trj.time, trj.unitcell_lengths, trj.unitcell_angles)

    # make new traj for middle chain 
    pos = []
    for ts in range(trj.n_frames):
        j = middle_chain_id[ts]
        sel = np.setdiff1d( np.arange(0,len(fasta)*chain_n), np.arange(len(fasta)*j,len(fasta)*(j+1)) )
        pos.append(trj.atom_slice(sel).xyz[ts])

    return np.array(pos)


# defines AH potential
HALR = lambda r,s,l : 4*0.8368*l*((s/r)**12-(s/r)**6)
HASR = lambda r,s,l : 4*0.8368*((s/r)**12-(s/r)**6)+0.8368*(1-l)
HA = lambda r,s,l : np.where(r<2**(1/6)*s, HASR(r,s,l), HALR(r,s,l))
HASP = lambda r,s,l,rc : np.where(r<rc, HA(r,s,l)-HA(rc,s,l), 0)

# define DH potential
temp = 293
ionic = 0.15 
RT = 8.3145*temp*1e-3
fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
epsw = fepsw(temp)
lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)

def contact_main(fasta_list,df,trj,prot1,prot2,prot1_n,prot2_n):
    """
    prot1: host protein
    prot2: guest protein
    """
    fasta1 = fasta_list[prot1]
    fasta2 = fasta_list[prot2]
    rc_coeff = 1.2
    sigmas,lambdas,qqs = read_sigma_lambda(df,fasta1,fasta2)

    # compute contact map
    cmap = np.zeros((trj.n_frames,len(fasta1)*len(fasta2)))
    ahmap = np.zeros((trj.n_frames,len(fasta1)*len(fasta2)))
    dhmap = np.zeros((trj.n_frames,len(fasta1)*len(fasta2)))
    offset = prot1_n*len(fasta1)

    for i in range(prot1_n):
        sel1 = np.arange(len(fasta1)*i,len(fasta1)*(i+1))
        sel2 = np.arange(len(fasta2)*0,len(fasta2)*1) + offset
        #print("prot1 {:d} to {:d}".format(sel1[0], sel1[-1]))
        #print("prot2 {:d} to {:d}".format(sel2[0], sel2[-1]))
        pairs_indices = trj.top.select_pairs(sel1,sel2)
        #print('pair indicies', pairs_indices)

        # "pairs" is used to save pairs (index,[res1,res2])
        if (i==0):
            pairs = pairs_indices

        d = md.compute_distances(trj,pairs_indices,periodic=True)
        cmap += np.where(d<rc_coeff*sigmas,1,0)
        ahmap += HASP(d,sigmas,lambdas,2.0)
        yukawa_eps = qqs*lB*RT
        dhmap += DHSP(d,yukawa_eps,lD,4.0)

        print("{:d}/{:d}".format(i+1,prot1_n))

    return pairs,cmap,ahmap,dhmap



def prepare(comp_list,comp_n_dic):

    # generate seq_len_list
    seq_len_dic = {}
    for comp_name in comp_list:
        seq_len_dic[comp_name] = get_seq_length(comp_name)
    print("seq_len_dic",seq_len_dic,"(number of beads)")
    # generate index list
    index_dic = get_index(comp_list,comp_n_dic,seq_len_dic,comp_list_sort=False)
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

    return name,comp_list,index_dic,seq_len_dic



def main(name, comp_list, index_dic, seq_len_dic, comp1_name, comp2_name, skip=1):
    """
    comp1: scaffold
    comp2: client (middle chain)
    """
    #print('This program calcaultes interaction of guest IDR at the middle of condensate (z~=0)')
    print("="*20+"start"+"="*20)
    print(f"system: {name}")
    fasta_list = read_fasta_to_dict('./fasta.fasta')
    #print('loaded fasta list (RNA is in 2-bead per nucleotide',fasta_list)
    df = pd.read_csv('./residues_C2RNA.csv').set_index('one')

    # extract prot1 and prot2  
    _ = md.load(f'traj/{name}.dcd', top=f'traj/{name}.pdb',stride=skip)
    _.xyz -= _.unitcell_lengths[0,:]/2 

    comp1_n = int( len(index_dic[comp1_name])/len(fasta_list[comp1_name]) )
    comp2_n = int( len(index_dic[comp2_name])/len(fasta_list[comp2_name]) )
    #print( len(index_dic[comp1_name]), len(fasta_list[comp1_name])  ) 
    print(f"interaction between {comp1_name} and {comp2_name}")  
    # extract the middle chains in comp2 (client)
    comp2_trj = _.atom_slice(index_dic[comp2_name])
    comp2_middle_trj_xyz,middle_chain_id = extract_middle_chain(df,fasta_list,comp2_trj,comp2_name,comp2_n)
 
    # extract comp1 (scaffold), and make topology N scaffold chains and 1 client chain 
    ## non-self interaction
    if comp1_name != comp2_name:
        comp1_trj =  _.atom_slice(index_dic[comp1_name])
        # top contains comp1 and comp2
        top = make_new_top(fasta_list,df,[[comp1_name,comp1_n],[comp2_name,1]])

    ## self interaction
    else:
        print(f'comp1 and comp2 are same, {comp1_name}')
        comp1_trj_tmp = _.atom_slice(index_dic[comp1_name])
        # non-middle chain is comp1 
        comp1_pos = extract_non_middle_chain(df,fasta_list,comp1_trj_tmp,comp1_name,comp1_n,middle_chain_id) 
        comp1_top = make_new_top(fasta_list,df,[[comp1_name,comp1_n-1]])
        comp1_trj = md.Trajectory(comp1_pos,comp1_top, _.time, _.unitcell_lengths, _.unitcell_angles)
        
        # top contains comp1 and comp2
        top = make_new_top(fasta_list,df,[[comp1_name,comp1_n]])

    print(top)
    print('{:s}-{:s} interaction is started to compute'.format(comp1_name,comp2_name))
    print(f'{comp1_name}: {comp1_n}, {comp2_name}: {comp2_n} interaction is started to compute (comp2 of middle chain is extracted)')

    # concatenate pos
    pos = np.concatenate([comp1_trj.xyz, comp2_middle_trj_xyz], axis=1)
    trj = md.Trajectory(pos, top, _.time, _.unitcell_lengths, _.unitcell_angles)
    print('comp1',comp1_trj.xyz.shape, '\ncomp2',comp2_middle_trj_xyz.shape, '\nare concatenated to', trj)
    #del _ 

    # Compute contact map
    if comp1_name != comp2_name:
        pairs,cmap,ahmap,dhmap = contact_main(fasta_list,df,trj,comp1_name,comp2_name,comp1_n,1) 
    else:
        pairs,cmap,ahmap,dhmap = contact_main(fasta_list,df,trj,comp1_name,comp1_name,comp1_n-1,1)

    np.save(f'data/mchain_pairs_{name}_{comp1_name}_{comp2_name}.npy', np.array(pairs))
    np.save(f'data/mchain_cmap_{name}_{comp1_name}_{comp2_name}.npy', cmap)
    np.save(f'data/mchain_ahmap_{name}_{comp1_name}_{comp2_name}.npy', ahmap)
    np.save(f'data/mchain_dhmap_{name}_{comp1_name}_{comp2_name}.npy', dhmap)
 
    print("="*100)

#========================================================================================-
#skip = 100 # 100 for test, 10 for production


# CAPRIN1_WT + ssRNA + dsRNA
#CAPRIN1_WT_ssRNA_dsRNA = True
if CAPRIN1_WT_ssRNA_dsRNA:
    comp1_name="CAPRIN1_N623TN630T"
    comp2_name='sspolyR12'
    comp3_name='dspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=15

    # define DH potential
    temp = 293.15
    ionic = 0.20
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSP_test", DHSP(1, -1*lB*RT, lD, 4))

    # some settings  
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)
    
    # CAPRIN1--dsRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_200mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--dsRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_200mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--dsRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_200mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

if CAPRIN1_WT_ssRNA_dsRNA_100mM:
    comp1_name="CAPRIN1_N623TN630T"
    comp2_name='sspolyR12'
    comp3_name='dspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=15

    # define DH potential
    temp = 293.15
    ionic = 0.08
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSP_test", DHSP(1, -1*lB*RT, lD, 4))

    # some settings  
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)

    # CAPRIN1--dsRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_100mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--dsRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_100mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--dsRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_100mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

if CAPRIN1_WT_ssRNA_dsRNA_80mM:
    comp1_name="CAPRIN1_N623TN630T"
    comp2_name='sspolyR12'
    comp3_name='dspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=15

    # define DH potential
    temp = 293.15
    ionic = 0.08
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSP_test", DHSP(1, -1*lB*RT, lD, 4))

    # some settings  
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)
    
    # CAPRIN1--dsRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--dsRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--dsRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)


# CAPRIN1_WT + ssRNA + label_ssRNA
#CAPRIN1_WT_ssRNA_label_ssRNA = True
if CAPRIN1_WT_ssRNA_label_ssRNA:
    comp1_name="CAPRIN1_N623TN630T"
    comp2_name='sspolyR12'
    comp3_name='label_sspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=30

    # define DH potential
    temp = 293.15
    ionic = 0.2
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSPtest", DHSP(1, -1*lB*RT, lD, 4))

    # some settings
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)

    # CAPRIN1--label_ssRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_200mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--label_ssRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_200mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--label_ssRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_200mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

if CAPRIN1_WT_ssRNA_label_ssRNA_100mM:
    comp1_name="CAPRIN1_N623TN630T"
    comp2_name='sspolyR12'
    comp3_name='label_sspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=30

    # define DH potential
    temp = 293.15
    ionic = 0.2
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSPtest", DHSP(1, -1*lB*RT, lD, 4))

    # some settings
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)

    # CAPRIN1--label_ssRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_100mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--label_ssRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_100mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--label_ssRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_100mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

if CAPRIN1_WT_ssRNA_label_ssRNA_80mM:
    comp1_name="CAPRIN1_N623TN630T"
    comp2_name='sspolyR12'
    comp3_name='label_sspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=30

    # define DH potential
    temp = 293.15
    ionic = 0.08
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSPtest", DHSP(1, -1*lB*RT, lD, 4))

    # some settings
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)

    # CAPRIN1--label_ssRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--label_ssRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--label_ssRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)




# CAPRIN1_RK + ssRNA + dsRNA
#CAPRIN1_RK_ssRNA_dsRNA = True
if CAPRIN1_RK_ssRNA_dsRNA:
    comp1_name="CAPRIN1_N623TN630T_RK"
    comp2_name='sspolyR12'
    comp3_name='dspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=15

    # define DH potential
    temp = 293.15
    ionic = 0.08
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSPtest", DHSP(1, -1*lB*RT, lD, 4))

    # some settings
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)

    # CAPRIN1_RK--dsRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--label_dsRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--label_dsRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)



# CAPRIN1_RK + ssRNA + label_ssRNA
#CAPRIN1_RK_ssRNA_label_ssRNA = True
if CAPRIN1_RK_ssRNA_label_ssRNA:
    comp1_name="CAPRIN1_N623TN630T_RK"
    comp2_name='sspolyR12'
    comp3_name='label_sspolyR12'
    comp1_n=500
    comp2_n=500
    comp3_n=15

    # define DH potential
    temp = 293.15
    ionic = 0.08
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.022/10)
    DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
    DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
    print("DHSPtest", DHSP(1, -1*lB*RT, lD, 4))

    # some settings
    comp_list = [comp1_name, comp2_name, comp3_name]
    comp_n_dic = {comp1_name:comp1_n, comp2_name:comp2_n, comp3_name:comp3_n}
    name,comp_list,index_dic,seq_len_dic = prepare(comp_list,comp_n_dic)

    # CAPRIN1_RK--label_ssRNA
    [scaff_name, client_name] = [comp1_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # ssRNA--label_ssRNA
    [scaff_name, client_name] = [comp2_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)

    # dsRNA--label_ssRNA
    [scaff_name, client_name] = [comp3_name, comp3_name]
    main(name+"_80mM", comp_list, index_dic, seq_len_dic, scaff_name, client_name, skip)


