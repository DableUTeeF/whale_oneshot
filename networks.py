import matching_networks
from torch.nn import functional as F
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.backends import cudnn
cudnn.benchmark = True
from keras.utils import data_utils
from torch.autograd import Variable


class Builder:
    def __init__(self, data, optim, lr):
        """
        Initializes the experiment
        :param data:
        """
        self.data = data
        self.classes_per_set = 5
        self.lr = lr
        self.image_size = (32, 32)
        self.optim = optim
        self.wd = 1e-6
        # self.g = matching_networks.Classifier(num_channels=3)
        self.g = matching_networks.ResNet()
        self.dn = matching_networks.DistanceNetwork()
        self.classify = matching_networks.AttentionalClassify()
        self.g_lstm = matching_networks.g_BidirectionalLSTM((32, 32, 512), 32, 1024, True).cuda()
        self.f_lstm = matching_networks.f_BidirectionalLSTM((64, 50, 50), 32, 1024, True).cuda()
        # self.matchNet = MatchingNetwork(keep_prob, batch_size, num_channels, self.lr, fce, classes_per_set,
        #                                 samples_per_class, image_size, self.isCuadAvailable & self.use_cuda)
        self.total_iter = 0
        self.g.cuda()
        self.total_train_iter = 0
        self.optimizer = self._create_optimizer(self.g, self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', verbose=True)

    def run_training_epoch(self, batch_size, num_worker):
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        traindata = self.data.get_trainset(batch_size, num_worker, shuffle=True)
        total_train_batches = len(traindata)
        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i, (x_support_set, y_support_set, x_target, y_target) in enumerate(traindata):
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                y_support_set = y_support_set.float()
                y_target = y_target.long()
                x_support_set = x_support_set.float()
                x_target = x_target.float()
                acc, c_loss = self.matchNet(x_support_set, y_support_set, x_target, y_target)

                # optimize process
                self.optimizer.zero_grad()
                c_loss.backward()
                self.optimizer.step()

                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                iter_out = f"loss: {total_c_loss / i:.{3}}, acc: {total_accuracy / i:.{3}}"
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1

            self.scheduler.step(total_accuracy)
            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            return total_c_loss, total_accuracy

    def matchNet(self, x_support_set, y_support_set_one_hot, x_target, y_target):
        # previously in matchnet.forward()
        encoded_images = []
        with torch.no_grad():
            for j in np.arange(x_support_set.size(1)):
                try:
                    gen_encode = self.g(x_support_set[:, j, :, :].cuda())
                    # gen_encode = self.g_lstm(gen_encode.unsqueeze(0)).squeeze(0).cpu()
                except RuntimeError as e:
                    raise RuntimeError(f'j={j}: {e}')
                encoded_images.append(gen_encode.cpu())
        f_encoded_image = self.g(x_target.cuda())
        output = torch.stack(encoded_images)
        # f_encoded_image = self.f_lstm(output.cuda(), f_encoded_image.cuda())
        # gen_encode = self.g_lstm(gen_encode.unsqueeze(0)).squeeze(0).cpu()
        # encoded_images.append(gen_encode.cpu())
        similarites = self.dn(support_set=output.cuda(), input_image=f_encoded_image.cuda())
        preds = self.classify(similarites, support_set_y=y_support_set_one_hot.cuda())
        values, indices = preds.max(1)
        acc = torch.mean((indices.squeeze() == y_target.cuda()).float())
        try:
            c_loss = F.cross_entropy(preds.cuda(), y_target.long().cuda())
        except RuntimeError as e:
            print(y_target.detach().numpy())
            # print(preds.cpu().detach().numpy())
            raise RuntimeError(e)
        return acc, c_loss

    def _create_optimizer(self, model, lr):
        # setup optimizer
        if self.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=self.wd)
        else:
            raise Exception("Not a valid optimizer offered: {0}".format(self.optim))
        return optimizer

    def run_val_epoch(self, batch_size, num_worker):
        total_c_loss = 0.0
        total_accuracy = 0.0
        valdata = self.data.get_testset(batch_size, num_worker)
        total_val_batches = len(valdata)

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i, (x_support_set, y_support_set, x_target, y_target) in enumerate(valdata):
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                y_support_set = y_support_set.float()
                y_target = y_target.long()
                x_support_set = x_support_set.float()
                x_target = x_target.float()
                with torch.no_grad():
                    acc, c_loss = self.matchNet(x_support_set, y_support_set, x_target, y_target)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                iter_out = f"v_loss: {total_c_loss / i:.{3}}, v_acc: {total_accuracy / i:.{3}}"
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_val_batches
            total_accuracy = total_accuracy / total_val_batches
            # self.scheduler.step(total_c_loss)
            return total_c_loss, total_accuracy

    def train_generator(self, batch_size, num_worker):
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        # traindata = self.data.get_trainset(batch_size, 0, shuffle=True)
        traindata = self.data
        train_enqueuer = data_utils.GeneratorEnqueuer(traindata)
        train_enqueuer.start(workers=num_worker, max_queue_size=batch_size*2)
        train_generator = train_enqueuer.get()
        total_train_batches = len(traindata)
        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                # (x_support_set, y_support_set, x_target, y_target) = next(train_generator)
                # batch = []
                # for b in range(batch_size):
                batch = next(train_generator)
                x_support_set, y_support_set, x_target, y_target = self.numpy2tensor(batch)
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                y_support_set = y_support_set.float()
                y_target = y_target.long()
                x_support_set = x_support_set.float()
                x_target = x_target.float()
                acc, c_loss = self.matchNet(x_support_set, y_support_set, x_target, y_target)

                # optimize process
                self.optimizer.zero_grad()
                c_loss.backward()
                self.optimizer.step()

                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                i += 1
                iter_out = f"loss: {total_c_loss / i:.{3}}, acc: {total_accuracy / i:.{3}}"
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1

            self.scheduler.step(total_accuracy)
            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            return total_c_loss, total_accuracy

    def validate_generator(self, batch_size, num_worker):
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        # traindata = self.data.get_trainset(batch_size, 0, shuffle=True)
        traindata = self.data
        val_enqueuer = data_utils.GeneratorEnqueuer(traindata)
        val_enqueuer.start(workers=num_worker, max_queue_size=batch_size*2)
        val_generator = val_enqueuer.get()
        total_val_batches = len(traindata)
        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):
                # (x_support_set, y_support_set, x_target, y_target) = next(train_generator)
                batch = []
                for b in range(batch_size):
                    batch.append(next(val_generator))
                x_support_set, y_support_set, x_target, y_target = self.numpy2tensor(batch)
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                y_support_set = y_support_set.float()
                y_target = y_target.long()
                x_support_set = x_support_set.float()
                x_target = x_target.float()
                with torch.no_grad():
                    acc, c_loss = self.matchNet(x_support_set, y_support_set, x_target, y_target)
                    total_c_loss += c_loss.data[0]
                    total_accuracy += acc.data[0]
                    iter_out = f"v_loss: {total_c_loss / i:.{3}}, v_acc: {total_accuracy / i:.{3}}"
                    pbar.set_description(iter_out)
                    pbar.update(1)
                    # self.total_train_iter+=1

                total_c_loss = total_c_loss / total_val_batches
                total_accuracy = total_accuracy / total_val_batches
                # self.scheduler.step(total_c_loss)
                return total_c_loss, total_accuracy

    def numpy2tensor(self, batch):
        x_support_set, y_support_set, x_target, y_target = batch
        x_s = Variable(torch.from_numpy(x_support_set)).float()
        y_s = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
        x_t = Variable(torch.from_numpy(x_target)).float()
        y_t = Variable(torch.from_numpy(y_target), requires_grad=False).long()
        return x_s, y_s, x_t, y_t
