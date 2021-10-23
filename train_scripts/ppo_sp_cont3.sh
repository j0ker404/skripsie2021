#!/bin/bash
#PBS -N slimePPOSPCont_Run_2000
#PBS -l select=1:ncpus=12:mem=64GB:ngpus=1:Qlist=ee
#PBS -l walltime=30:00:00
#PBS -m ae
#PBS -e outSP_train_cont_2000.err
#PBS -o outSP_train_cont_2000.out
#PBS -M 21733902@sun.ac.za

# make sure I'm the only one that can read my output
umask 0077
# create a temporary directory with the job ID as name in /scratch-small-local
SPACED="${PBS_JOBID//./-}" 
# path for comp047
TMP=/scratch-large-network/${SPACED} # E.g. 249926.hpc1.hpc
mkdir -p ${TMP}
echo "Temporary work dir: ${TMP}"

cd ${TMP}


# load python3
module load python/3.8.1

# create virtual venv
python3 -m venv venv

# Ensure virtual venv activated
echo "Activating venv"
source venv/bin/activate


# copy the input files to ${TMP}
echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
/usr/bin/rsync -vax "${PBS_O_WORKDIR}/" ${TMP}/


# You may need to add additional lines here if your script
# requires custom git / pip dependancies not included in your conda env.
pip install -r requirements.txt 
# install pytorch
pip install torch torchvision torchaudio

echo "Start Training"
python ./train/selfplay/policygrad/train_cont.py
echo "Stop Training"

# job done, copy everything back
echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/"
/usr/bin/rsync -vax ${TMP}/ "${PBS_O_WORKDIR}/"

# if the copy back succeeded, delete my temporary files
cd ..
[ $? -eq 0 ] && /bin/rm -rf ${TMP}