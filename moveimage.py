import os
import shutil

csv = open('misc/train.csv', 'r').readlines()
lists = []
train_list = {}
test_list = {}
all_list = {}
for line in csv[1:]:
    id_, target = line[:-1].split(',')
    lists.append((id_, target))

    if target not in test_list:
        test_list[target] = [id_]
    else:
        if target not in train_list:
            train_list[target] = []
        train_list[target].append(id_)
    if target not in all_list:
        all_list[target] = [id_]
    else:
        all_list[target].append(id_)

lslength = len(lists)
ratio = 0.8
# train_list = lists[:int(lslength * 0.8)]
# test_list = lists[int(lslength * 0.8):]
dests = ['train']
lists = [all_list]
try:
    os.listdir('/root')
    rootpath = '/root/palm/DATA/whale/2to7'
    sourcepath = '/root/palm/DATA/whale/train'
except PermissionError:
    rootpath = '/media/palm/data/whale/2to7'
    sourcepath = '/media/palm/data/whale/train'
if not os.path.isdir(rootpath):
    os.mkdir(rootpath)
for i in range(1):
    if not os.path.isdir(os.path.join(rootpath, dests[i])):
        os.mkdir(os.path.join(rootpath, dests[i]))
    x = 0
    for key in lists[i]:
        # if key == 'new_whale':
        #     continue
        if len(lists[i][key]) > 7 or len(lists[i][key]) < 2:
            continue
        for idx, item in enumerate(lists[i][key]):
            if idx > 100:
                break
            if not os.path.isdir(os.path.join(rootpath, dests[i], key)):
                os.mkdir(os.path.join(rootpath, dests[i], key))
            source = os.path.join(sourcepath, item)
            destination = os.path.join(rootpath, dests[i], key, item)
            shutil.copy(source, destination)
