from datagen import Generator
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
name = 'morethan7/train'
# rootpath = '/media/palm/PyCharmProjects/DATA/cat_vs_dog'
# name = 'train_val'

data = Generator(os.path.join(rootpath, name))
obj_oneShotBuilder = Builder(data, 'sgd', 0.01)

for e in range(total_epochs):
    total_c_loss, total_accuracy = obj_oneShotBuilder.train_generator(batch_size, 2)
    print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
    # total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.validate_generator(batch_size, 1)
    # print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
    # if total_val_accuracy > best_val_acc:
    #     best_val_acc = total_val_accuracy
