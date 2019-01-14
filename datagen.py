from PIL import Image
import numpy as np
import os
import warnings


class Generator:
    def __init__(self, path, csv=None, batch_size=8, imsize=(224, 224), im_per_cls=1, cls_per_batch=100):
        self.path = path
        self.csv = csv
        self.curidx = 0
        self.batch_size = batch_size
        self.imsize = imsize
        self.im_per_cls = im_per_cls  # still not available due to class selection in getitem
        self.im_per_cls = 1
        self.cls_per_batch = cls_per_batch

        if csv is not None:
            self.init_csv()
        else:
            self.init_dir()
        self.imidx = list(self.dataset)

    def init_dir(self):
        # load images
        self.dataset = {}
        self.class_list = {}
        folders = os.listdir(self.path)
        i = 0
        for folder in folders:
            for imname in os.listdir(os.path.join(self.path, folder)):
                i += 1
                print(i, end='\r')
                try:
                    img = Image.open(os.path.join(self.path, folder, imname)).resize(self.imsize).convert('RGB')
                    self.dataset[imname] = np.array(img)
                    self.class_list[imname] = folder
                except OSError:
                    pass

    def init_csv(self):
        # load images
        self.dataset = {}
        for imname in os.listdir(os.path.join(self.path)):
            try:
                img = Image.open(os.path.join(self.path, imname)).resize(self.imsize).convert('RGB')
                self.dataset[imname] = np.array(img)
            except OSError:
                warnings.warn('There are some non-image file(s)')

        # create class list
        self.class_list = {}
        for line in self.csv:
            if line.startswith('Image'):
                continue
            if line.endswith('\n'):
                line = line[:-1]
            ls = line.split(',')
            self.class_list[ls[0]] = ls[1]

    def __len__(self):
        return len(self.dataset)//self.batch_size

    def __getitem__(self, idx):
        x_support = np.zeros((self.batch_size, self.cls_per_batch, self.im_per_cls, self.imsize[1], self.imsize[0], 3), dtype='uint8')
        y_support = np.zeros((self.batch_size, self.cls_per_batch, self.im_per_cls, self.cls_per_batch), dtype='uint8')
        x_target = np.zeros((self.batch_size, self.imsize[1], self.imsize[0], 3), dtype='uint8')
        y_target = np.zeros((self.batch_size, 1), dtype='uint8')
        for b in range(self.batch_size):
            # support set
            selected_idxs = []
            seen_cls = []
            for i in range(self.cls_per_batch):
                for k in range(self.im_per_cls):
                    randomed_idx = np.random.randint(0, len(self.dataset) - 1)
                    randomed_cls = self.class_list[self.imidx[randomed_idx]]
                    while randomed_idx in selected_idxs or randomed_cls in seen_cls or randomed_cls == 'new_whale':
                        randomed_idx = np.random.randint(0, len(self.dataset) - 1)
                    selected_idxs.append(randomed_idx)
                    seen_cls.append(randomed_cls)
                    x = self.dataset[self.imidx[randomed_idx]]
                    y = len(seen_cls)-1
                    x_support[b, i, k] = np.array(x, dtype='uint8')
                    y_support[b, i, k, y] = 1

            # target set
            randomed_cls = self.class_list[self.imidx[idx]]
            while idx in selected_idxs or randomed_cls not in seen_cls:
                idx = np.random.randint(0, len(self.dataset) - 1)
            x = self.dataset[self.imidx[idx]]
            y = seen_cls.index(self.class_list[self.imidx[idx]])
            x_target[b] = x
            y_target[b] = y

        s = x_support.shape
        x_support = np.reshape(x_support, (s[0], s[1]*s[2], s[3], s[4], s[5]))
        s = y_support.shape
        y_support = np.reshape(y_support, (s[0], s[1]*s[2], s[3]))
        return x_support, y_support, x_target, y_target

    def __next__(self):
        self.curidx += 1
        return self[self.curidx-1]
