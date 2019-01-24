from datagen import Generator
from networks import Builder
import os
import warnings
warnings.simplefilter("ignore")
# Experiment setup
# todo: Just store every image as a PIL image in a list, it would use less ram and run faster
batch_size = 8
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
train_name = 'morethan7/train'
val_name = '2to7/train'
weightspath = 'checkpoints/resnet-1.h5'

train_data = Generator(os.path.join(rootpath, train_name), batch_size=batch_size)
obj_oneShotBuilder = Builder('sgd', 0.001)
val_data = Generator(os.path.join(rootpath, val_name), batch_size=batch_size)

obj_oneShotBuilder.load_weights('checkpoints/tmp.h5')

for e in range(total_epochs):
    print('Epoch: {}'.format(e+1))
    total_c_loss, total_accuracy = obj_oneShotBuilder.train_generator(train_data, 2)
    total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.validate_generator(val_data, 2)
    print(f"train_loss: {total_c_loss:.4} train_acc: {total_accuracy:.4} val_loss: {total_val_c_loss:.4} val_acc: {total_val_accuracy: .3}")
    if total_val_accuracy > best_val_acc:
        best_val_acc = total_val_accuracy
        obj_oneShotBuilder.save_weights(weightspath)
    obj_oneShotBuilder.save_weights(weightspath+'-tmp')
