from data_loader import FolderDatasetLoader
from Model import OmniglotBuilder
import tqdm
import os
# Experiment setup
batch_size = 16
fce = True
classes_per_set = 5005
classes_per_set = 2
samples_per_class = 3
channels = 1
# Training setup
total_epochs = 100
total_train_batches = 1000
total_val_batches = 250
total_test_batches = 500
best_val_acc = 0.0
# try:
#     os.listdir('/root')
#     rootpath = '/root/palm/DATA/whale'
# except PermissionError:
#     rootpath = '/media/palm/data/whale'
name = 'newwhaled'
rootpath = '/media/palm/PyCharmProjects/DATA/cat_vs_dog'
name = 'train_val'

data = FolderDatasetLoader(num_of_gpus=1, batch_size=batch_size, image_height=28, image_width=28,
                           image_channels=3,
                           train_val_test_split=(0.7, 0.2, 0.1),
                           samples_per_iter=1, num_workers=4,
                           data_path="datasets/cat-dog", name="cat-dog",
                           indexes_of_folders_indicating_class=[-2, -3], reset_stored_filepaths=False,
                           num_samples_per_class=samples_per_class,
                           num_classes_per_set=classes_per_set, label_as_int=False)
obj_oneShotBuilder = OmniglotBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size=batch_size, num_channels=3, lr=1e-3, image_size=28, classes_per_set=classes_per_set,
                                    samples_per_class=1, keep_prob=0.0, fce=True, optim="adam", weight_decay=0,
                                    use_cuda=True)

with tqdm.tqdm(total=total_train_batches) as pbar_e:
    for e in range(total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(total_val_batches)
        print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
        if total_val_accuracy > best_val_acc:
            best_val_acc = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_test_epoch(total_test_batches)
            print("Epoch {}: test_loss:{} test_accuracy:{}".format(e, total_test_c_loss, total_test_accuracy))
        pbar_e.update(1)
