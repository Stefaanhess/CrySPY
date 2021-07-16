#! /bin/bash
#$ -cwd
#$ -t 1-1
#$ -V
#$ -S /bin/bash
#$ -l m_core=10
#$ -e /home/stefaah94/logs/
#$ -o /home/stefaah94/logs/
#$ -r n

export TMP_DATA_DIR=/tmp/$USER/$JOB_ID.$SGE_TASK_ID
export DB_NAME=train.db
mkdir $TMP_DATA_DIR

cp data/$DB_NAME $TMP_DATA_DIR/custom.db

spktrain +experiment=pes data.datapath=$TMP_DATA_DIR/custom.db trainer.gpus=1 model/representation=painn model.representation.n_atom_basis=64 model.representation.n_interactions=3 lr=0.0005 cutoff=5 run_id="job$1_$2" data.num_train=10 data.num_val=10 callbacks.early_stopping.patience=1

rm -rf $TMP_DATA_DIR
rm "lock_train_$1_$2"

