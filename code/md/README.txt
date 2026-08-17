
#ddx4
rsync -avr --exclude='DDX4'   ../../../md/DDX4-350/ ./

#ddx4_ssnra
rsync -avr  --exclude='md'  --exclude='*.log'   --exclude='*.out'  --exclude='*.slurm'  --exclude='*.sh'  --exclude='*.err'  ../../../md/slab_DDX4_ssACUG24_grid/* ./

#ddx4_dsrna
rsync -avr  --exclude='md'  --exclude='*.log'   --exclude='*.out'  --exclude='*.slurm'  --exclude='*.sh'  --exclude='*.err'  ../../../md/slab_DDX4_dsACUG24_grid_update/* ./

#ddx4_dsdna
rsync -avr  --exclude='md'  --exclude='*.log'  --exclude='log.run1'  --exclude='*.out'  --exclude='*.slurm'  --exclude='*.sh'  --exclude='*.err'  ../../../dna_md/slab_DDX4_dsACTG24* ./

#caprin1
rsync -avr  --exclude='md'  --exclude='*.log'  --exclude='log.run1'  --exclude='*.out'  --exclude='*.slurm'  --exclude='*.sh'  --exclude='*.err' --exclude='log.*'  ../../../caprin_md/*  ./

#280bp dsRNA
rsync -avr  --exclude='md'  --exclude='*.log'  --exclude='log.run1'  --exclude='*.out'  --exclude='*.slurm'  --exclude='*.sh'  --exclude='*.err' --exclude='log.*'    ../../../md/single_dsACUG_stiff_grid_l013_update/single_dsACUG_n136*  ./

#280bp dsDNA
rsync -avr  --exclude='md'  --exclude='*.log'  --exclude='log.run1'  --exclude='*.out'  --exclude='*.slurm'  --exclude='*.sh'  --exclude='*.err' --exclude='log.*'    ../../../dna_md/single_dsACTG_n136_*_l065  ./
