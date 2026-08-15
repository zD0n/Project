#!/bin/bash
#SBATCH --job-name='LEAF_Prj'
#SBATCH --chdir=/nfs-share-stgnode/home/663380260-7/Workshop
#SBATCH --partition=GPU_FOR_AI_STU
#SBATCH --nodes=1
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Submit with:   sbatch job.sbatch
# Watch with:    squeue -u $USER      /      tail -f slurm-<jobid>.out
#
# NOTE: --exclusive was dropped. It reserves the whole node for one GPU job and
# usually means a much longer queue wait. Add it back if your admins require it.

export SLURM_OVERLAP=yes

WORKDIR=/nfs-share-stgnode/home/663380260-7/Workshop
SIF=/nfs-share-stgnode/containers/pytorch_25_04_py3.sif

# Which experiment this job runs. Run5 and Run8 are pure PyTorch and work in the
# stock PyTorch container above. Run6/Run7 need TensorFlow and will stop with a
# clear message if the container lacks it -- for those, point SIF at an image
# built from leaf_vit_h100.def.
#
# Run8 with EXTRA="MODEL=convnext" also needs timm; run_server.sh installs it
# into ~/.local on first use if the container has no timm, so the stock
# container is enough.
DATASET=iemocap
SCRIPT=Run5.py
EXTRA=""            # e.g. EXTRA="MODEL=convnext SEED=42 NUM_EPOCHS=30"

module try-load singularity

echo "job start time is $(date)"
echo "node: ${SLURM_JOB_NODELIST}   job: ${SLURM_JOB_ID}"

exec_node=${SLURM_JOB_NODELIST}
cpu_curr_node=${SLURM_CPUS_ON_NODE}
if [ "${cpu_curr_node}" = "" ]; then cpu_curr_node=4; fi

set -x
srun -N1 -n1 --cpu_bind=cores -l --nodelist=${exec_node} \
    --cpus-per-task=${cpu_curr_node} \
    singularity exec --nv \
    -B /nfs-share-stgnode/home/663380260-7 \
    --pwd ${WORKDIR} \
    ${SIF} bash ${WORKDIR}/run_server.sh ${DATASET} ${SCRIPT} ${EXTRA}
set +x

echo "job end time is $(date)"
