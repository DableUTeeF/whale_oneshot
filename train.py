from datagen import Generator
from networks import Builder
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
weightspath = 'checkpoints/resnet-1.h5'

data = Generator(os.path.join(rootpath, name))
obj_oneShotBuilder = Builder(data, 'sgd', 0.001)

for e in range(total_epochs):
    total_c_loss, total_accuracy = obj_oneShotBuilder.train_generator(batch_size, 2)
    print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
    obj_oneShotBuilder.save_weights(weightspath)
