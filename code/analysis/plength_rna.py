import sys
sys.path.append('BLOCKING')
from main import BlockAnalysis
import mdtraj as md
import numpy as np


def main(stack_n,krstr,path='../md/single_dsACUG_stiff_grid_l014/',seqname='dsACUG'):  #def main(krstr,key=''):
    name = 'md' #'acug'
    dir_= path + f'single_{seqname}_n{stack_n}_k{krstr}/{name}/'
    b = 100 # 100 ns
    trj = md.load(dir_+f'{name}.dcd', top=dir_+f'top.pdb')[b:]
    print("loaded traj", dir_+f'{name}.dcd')

    # mid-point  
    # select P atom in chain A, within residues from 5 to (n-5)-th
    p_index_A = trj.topology.select("(name sP) and (resid < 260) and (resid >= 20)")
    n_index_A = trj.topology.select("(name sN) and (resid < 260) and (resid >= 20)")
    # select P atom in chain B, within residues from 5 to (n-5)-th
    p_index_B = trj.topology.select("(name sP) and (resid >= 300) and (resid < 540)")
    n_index_B = trj.topology.select("(name sN) and (resid >= 300) and (resid < 540)")

    p_pos_A = trj.xyz[:, p_index_A, :]   # shape: (n_frames, n_atoms_A, 3)
    n_pos_A = trj.xyz[:, n_index_A, :]   # shape: (n_frames, n_atoms_A, 3)
    p_pos_B = trj.xyz[:, p_index_B, :]   # shape: (n_frames, n_atoms_B, 3)
    n_pos_B = trj.xyz[:, n_index_B, :]   # shape: (n_frames, n_atoms_B, 3)
    # change order of chain B
    p_pos_B_rev = p_pos_B[:, ::-1, :]
    n_pos_B_rev = n_pos_B[:, ::-1, :]
  
    pos_mid = (p_pos_A + n_pos_A + p_pos_B_rev + n_pos_B_rev) / 4.0 
    n_bp = p_pos_A.shape[1]
    print(f"Base pairs counted: {n_bp}")

    # per turn
    index_per_turn = np.arange(0, 200, 11)
    pos_per_turn = pos_mid[:, index_per_turn, :]
    print("sampling distance in unit of base", index_per_turn)
    #print(  pos_per_turn.shape)

    # compute rise per turn 
    l0 =    np.mean( np.linalg.norm(pos_per_turn[:,1:,:]-pos_per_turn[:,:-1,:], axis=-1) )
    l0_std = np.std( np.linalg.norm(pos_per_turn[:,1:,:]-pos_per_turn[:,:-1,:], axis=-1) )
    print(f"rise per turn: {str(l0)[:4]} \pm {str(l0_std)[:4]}")

    # auto-correlation 
    ln_vect = pos_per_turn[:,1:,:] -pos_per_turn[:,:-1,:]   # vector from i to i+1
    l0_vect = pos_per_turn[:,1:2,:]-pos_per_turn[:,0:1,:]  # vector from 0 to 1
    ln_l0 = np.sum(ln_vect*l0_vect, axis=-1)     # dot product of ln and l0
    
    ln_l0_ts = ln_l0                             # time-series 

    ln_l0 = np.mean(ln_l0,axis=0)                # time-average
    ln_l0_norm = ln_l0/ln_l0[0]
    print("auto correlation function", ln_l0_norm)

    # fit autocorrelation function into expopentiral decay
    #xs = np.arange(len(ln_l0_norm)) * l0 
    xs = np.arange(0, len(ln_l0_norm)) * l0  #xs = index_per_turn[1:]  * l0 
    a = np.sum( xs * np.log(ln_l0_norm) ) / np.sum(xs**2)
    l_ps = -1/a


    # estimate standard deviation of mean 
    l_ps_list = []
    ts = 380  # number of steps 
    print(f"time blocks: N_blocls {len(ln_l0_ts)//ts}, Steps per block {ts}")
    for i in range( len(ln_l0_ts)//ts ):
        ln_l0_i = ln_l0_ts[i*ts:(i+1)*ts].mean(axis=0)
        ln_l0_i_norm = ln_l0_i/ln_l0[0]
        a = np.sum( xs * np.log(ln_l0_i_norm) ) / np.sum(xs**2)        
        l_ps_i = -1/a
        l_ps_list.append(l_ps_i)
    l_ps_sem = np.std(np.array(l_ps_list))/np.sqrt(len(l_ps_list))
   
    print(f'p length {l_ps} \pm {l_ps_sem} [nm]') 

    np.save(f'./data/single_{seqname}_n{stack_n}_k{krstr}_corr.npy',ln_l0_norm)
    np.savez(f'./data/single_{seqname}_n{stack_n}_k{krstr}_plength.npz',mean=l_ps,sem=l_ps_sem)
    print("="*100)


#stack_n = 15
#for krstr in [8,10,20]:
#    main(stack_n,krstr,path='../md/single_dsACUG_stiff_grid_l013_update/')


stack_n = 136
for krstr in [8,10,20]:
    main(stack_n,krstr,path='../md/single_dsACUG_stiff_grid_l013_update/')

