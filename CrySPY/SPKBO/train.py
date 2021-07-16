import subprocess
from ..IO import change_input, io_stat, pkl_data
import os


def train_models(n_models):
    best_value, bo_epoch, training_epoch = pkl_data.load_spkbo_data()
    training_epoch += 1
    pkl_data.save_spkbo_data([best_value, bo_epoch, training_epoch])
    for i in range(n_models):
        print(f"submitted training job for model {i} in bo-epoch {training_epoch}")
        with open(f"lock_train_{training_epoch}_{i}", "w") as file:
            file.write("lock file for indicating training - will be removed when training is done")
        subprocess.Popen("pwd")
        subprocess.Popen(["qsub", "/home/stefaah94/Software/CrySPY/CrySPY/SPKBO/spktrain.sh", str(training_epoch), str(i)])


def training_still_running():
    return len([f for f in os.listdir(".") if f.startswith("lock_train")]) > 0
