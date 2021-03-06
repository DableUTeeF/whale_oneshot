from data_loader import FolderDatasetLoader
from networks import Builder
import tqdm
import os
import warnings
warnings.simplefilter("ignore")
# Experiment setup
# todo: Just store every image as a PIL image in a list, it would use less ram and run faster
batch_size = 32
fce = True
classes_per_set = 32
# classes_per_set = 2
samples_per_class = 1
channels = 1
# Training setup
total_epochs = 100
total_train_batches = 1000
total_val_batches = 250
total_test_batches = 500
best_val_acc = 0.0
try:
    os.listdir('/root')
    rootpath = '/root/palm/DATA/whale'
except PermissionError:
    rootpath = '/media/palm/data/whale'
name = 'morethan7'
# rootpath = '/media/palm/PyCharmProjects/DATA/cat_vs_dog'
# name = 'train_val'

data = FolderDatasetLoader(num_of_gpus=1, batch_size=batch_size, image_height=224, image_width=512,
                           image_channels=3,
                           train_val_test_split=(0.7, 0.2, 0.1),
                           samples_per_iter=1, num_workers=0,
                           data_path=rootpath, name=name,
                           indexes_of_folders_indicating_class=[-2, -3], reset_stored_filepaths=False,
                           num_samples_per_class=samples_per_class,
                           num_classes_per_set=classes_per_set, label_as_int=False, seed=2007)
obj_oneShotBuilder = Builder(data, 'sgd', 0.01)

with tqdm.tqdm(total=total_train_batches) as pbar_e:
    for e in range(total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.train_generator(batch_size, 1)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(batch_size, 1)
        print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
        if total_val_accuracy > best_val_acc:
            best_val_acc = total_val_accuracy
        pbar_e.update(1)
