import numpy as np

ddx4_only = True
ddx4_24bp_ssRNA_small = True
ddx4_24bp_ssRNA_large = True
ddx4_24bp_ssRNA_large_ssrna_8 = True
ddx4_24bp_dsRNA_small = True
ddx4_24bp_dsRNA_large = True
ddx4_24bp_dsRNA_large_flexible = True

CAPRIN1_WT_ssRNA_dsRNA = True
CAPRIN_WT_ssRNA_label =  True
CAPRIN1_WT_ssRNA_dsRNA_80mM = True
CAPRIN_WT_ssRNA_label_80mM =  True
CAPRIN1_WT_ssRNA_dsRNA_100mM = True
CAPRIN_WT_ssRNA_label_100mM =  True
CAPRIN1_RK_ssRNA_dsRNA = True
CAPRIN_RK_ssRNA_label =  True

ddx4_24bp_dsDNA = True



def ddx4_plot_concentration(name, comp_name, path):
    print("loading", path+name)
    # load density profile. this is time-series 
    h = np.load(path+"hist_"+name+"_"+comp_name+'.npy')
    x = np.arange(0,len(h.mean(axis=0)))   # bin size 1 nm
    x = x-np.mean(x)                        # centering z=0

    # load boundary line   
    bb = np.load(path+"cutoff_"+name+".npy") # [dens_pos, dens_neg, dil_pos, dil_neg]
    print("bb", bb)
    x_index_den = np.logical_and(bb[1]<x, x < bb[0]) 
    print("x_index_dens", x_index_den)
    x_index_dil = np.logical_or(x<bb[-1], bb[-2]<x )
    print("x_index_dil", x_index_dil)
    print("==="*20)
    
    # compute concentration
    h_den = h[:, x_index_den]
    conc_den_for_frame = h_den.mean(axis=1)
    h_dil = h[:, x_index_dil]
    conc_dil_for_frame = h_dil.mean(axis=1)
    
    return conc_den_for_frame, conc_dil_for_frame



def main(name, comp_list, path):
    for comp in comp_list:
        dens, dil = ddx4_plot_concentration(name, comp, path=path)
        np.save(path+f"dense_conc_frame_{name}_{comp}.npy", dens)
        np.save(path+f"dilute_conc_frame_{name}_{comp}.npy", dil)

    
##########################################################################

path = './data/'

# DDX4 only system
if ddx4_only:
    comp_list = ["DDX4"]
    main("DDX4-350", comp_list, path)

# small DDX4-RNA systems (400 chains)
if ddx4_24bp_ssRNA_small:
    name_list = ["DDX4-400_ss_acug24-8", 
                 "DDX4-400_ss_acug24-16", 
                 "DDX4-400_ss_acug24-24", 
                 "DDX4-400_ss_acug24-32" ]
    comp_list = ["DDX4", "ss_acug24"]

    for name in name_list:
        main(name, comp_list, path)


# large DDX4-RNA systems (800 chains), ssRNA 16 chains
if ddx4_24bp_ssRNA_large:
    name_list = ["DDX4-800_ss_acug24-16_1", 
                 "DDX4-800_ss_acug24-16_2", 
                 "DDX4-800_ss_acug24-16_3", 
                 "DDX4-800_ss_acug24-16_4", 
                 "DDX4-800_ss_acug24-16_5", ]
    comp_list = ["DDX4", "ss_acug24"]

    for name in name_list:
        main(name, comp_list, path)


# large DDX4-RNA systems (800 chains), ssRNA 8 chains
if ddx4_24bp_ssRNA_large_ssrna_8:
    name_list = ["DDX4-800_ss_acug24-8_1",
                 "DDX4-800_ss_acug24-8_2",
                 "DDX4-800_ss_acug24-8_3",
                 "DDX4-800_ss_acug24-8_4" ]
    comp_list = ["DDX4", "ss_acug24"]

    for name in name_list:
        main(name, comp_list, path) 


# small DDX4-dsRNA systems (300 chains)
if ddx4_only:
    name_list = ["DDX4-300_ds_acug24-15_direct_l00",
                 "DDX4-300_ds_acug24-15_direct_l01",
                 "DDX4-300_ds_acug24-15_direct_l02",
                 "DDX4-300_ds_acug24-15_direct_l03"]
    comp_list = ["DDX4", "ds_acug24"]

    for name in name_list:
        main(name, comp_list, path)


# large DDX4-dsRNA systems (400 chains)
if ddx4_24bp_dsRNA_small:
    name_list = ["DDX4-400_ds_acug24-15_direct_l00",
                 "DDX4-400_ds_acug24-15_direct_l01",
                 "DDX4-400_ds_acug24-15_direct_l013",
                 "DDX4-400_ds_acug24-15_direct_l02",
                 "DDX4-400_ds_acug24-15_direct_l03"]
    comp_list = ["DDX4", "ds_acug24"]

    for name in name_list:
        main(name, comp_list, path)



# large DDX4-dsRNA systems (400 chains)
# 24 bps RNA in doudle strand "flexible"
if ddx4_24bp_dsRNA_large:
    name_list = ["DDX4-400_ds_acug24-15_direct_l01_k10",
                 "DDX4-400_ds_acug24-15_direct_l013_k10",
                 "DDX4-400_ds_acug24-15_direct_l02_k10"]
    comp_list = ["DDX4", "ds_acug24"]

    for name in name_list:
        main(name, comp_list, path)



# CAPRIN1_WT + ssRNA + dsRNA
if CAPRIN1_WT_ssRNA_dsRNA:
    name = 'CAPRIN1_N623TN630T-500_sspolyR12-500_dspolyR12-15_200mM'
    comp_list= ['CAPRIN1_N623TN630T', 'sspolyR12', 'dspolyR12']
    main(name, comp_list, path)

if CAPRIN1_WT_ssRNA_dsRNA_80mM:
    name = 'CAPRIN1_N623TN630T-500_sspolyR12-500_dspolyR12-15_80mM'
    comp_list= ['CAPRIN1_N623TN630T', 'sspolyR12', 'dspolyR12']
    main(name, comp_list, path)

if CAPRIN1_WT_ssRNA_dsRNA_100mM:
    name = 'CAPRIN1_N623TN630T-500_sspolyR12-500_dspolyR12-15_100mM'
    comp_list= ['CAPRIN1_N623TN630T', 'sspolyR12', 'dspolyR12']
    main(name, comp_list, path)

# CAPRIN1_WT + ssRNA + labeled_ssRNA
if CAPRIN_WT_ssRNA_label:
    name = 'CAPRIN1_N623TN630T-500_sspolyR12-500_label_sspolyR12-30_200mM'
    comp_list= ['CAPRIN1_N623TN630T', 'sspolyR12', 'label_sspolyR12']
    main(name, comp_list, path)

if CAPRIN_WT_ssRNA_label_80mM:
    name = 'CAPRIN1_N623TN630T-500_sspolyR12-500_label_sspolyR12-30_80mM'
    comp_list= ['CAPRIN1_N623TN630T', 'sspolyR12', 'label_sspolyR12']
    main(name, comp_list, path)

if CAPRIN_WT_ssRNA_label_100mM:
    name = 'CAPRIN1_N623TN630T-500_sspolyR12-500_label_sspolyR12-30_100mM'
    comp_list= ['CAPRIN1_N623TN630T', 'sspolyR12', 'label_sspolyR12']
    main(name, comp_list, path)

# CAPRIN1_RK + ssRNA + dsRNA
if CAPRIN1_RK_ssRNA_dsRNA:
    name = 'CAPRIN1_N623TN630T_RK-500_sspolyR12-500_dspolyR12-15_80mM'
    comp_list= ['CAPRIN1_N623TN630T_RK', 'sspolyR12', 'dspolyR12']
    main(name, comp_list, path)


# CAPRIN1_RK + ssRNA + labeled_ssRNA
if CAPRIN_RK_ssRNA_label:
    name = 'CAPRIN1_N623TN630T_RK-500_sspolyR12-500_label_sspolyR12-30_80mM'
    comp_list= ['CAPRIN1_N623TN630T_RK', 'sspolyR12', 'label_sspolyR12']
    main(name, comp_list, path)


# ddx4_24bp_dsDNA:
if ddx4_24bp_dsDNA:
    name_list = ["DDX4-400_dsACTG24-15",
                 "DDX4-400_dsACTG24-15_l02",
                 "DDX4-400_dsACTG24-15_l03",
                 "DDX4-400_dsACTG24-15_l04",
                 "DDX4-400_dsACTG24-15_l06",
                 "DDX4-400_dsACTG24-15_l065",
                 "DDX4-400_dsACTG24-15_l07",
                 "DDX4-400_dsACTG24-15_l08"]
    comp_list = ["DDX4", "dsACTG24"]

    for name in name_list:
        main(name, comp_list, path)















