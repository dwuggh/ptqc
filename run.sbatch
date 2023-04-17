#!/bin/bash
#SBATCH -J ptqc
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p CPU-Small
#SBATCH --qos=scpujoblimit
#SBATCH --time=120:00:00
#SBATCH -N 1
#SBATCH -n 40

# cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate quantum

for n in {4..16..2}; do
    for p in {05..35..5}; do
	# echo "$n-$p"
        ( python main.py n 0.0$p & )
    done
done

