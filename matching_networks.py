import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last_relu, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.last_relu:
            x = F.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, False)
        self.res1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = F.max_pool2d(y, (3, 3), (2, 2), (1, 1))
        x = self.res1(x)
        y += x
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out += self.shortcut(x)
        out = out
        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(3, 4, 6, 3)):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, 1024)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


class Classifier(nn.Module):
    def __init__(self, num_channels=1):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.conv1 = ConvBlock(num_channels, 32, True, stride=1)
        self.res1 = ResBlock(32, 64)
        self.res2 = ResBlock(64, 128)
        self.res3 = ResBlock(128, 256)

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.conv1(image_input)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = x.view(x.size()[0], -1)
        return x


class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


class g_BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim, use_cuda):
        super(g_BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self, use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                             requires_grad=False).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                             requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                             requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                             requires_grad=False))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        # self.hidden = self.init_hidden(self.use_cuda)
        # self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class f_BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim, use_cuda, k=10):
        super(f_BidirectionalLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)
        self.k = k
        self.attentional_softmax = nn.Linear(vector_dim, vector_dim)
        self.reuse = False

    def init_hidden(self, use_cuda):
        if use_cuda:
            return (
            Variable(torch.zeros(self.lstm.num_layers * 2, 1, self.lstm.hidden_size), requires_grad=False).cuda(),
            Variable(torch.zeros(self.lstm.num_layers * 2, 1, self.lstm.hidden_size), requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, 1, self.lstm.hidden_size), requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, 1, self.lstm.hidden_size), requires_grad=False))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, support_set_embeddings, target_set_embeddings, K=None):
        """"""
        '''
        def __call__(self, support_set_embeddings, target_set_embeddings, K, training=False):
            b, k, h_g_dim = support_set_embeddings.get_shape().as_list()
            b, h_f_dim = target_set_embeddings.get_shape().as_list()
            with tf.variable_scope(self.name, reuse=self.reuse):
                fw_lstm_cells_encoder = rnn.LSTMCell(num_units=self.layer_size, activation=tf.nn.tanh)
                attentional_softmax = tf.ones(shape=(b, k)) * (1.0/k)
                h = tf.zeros(shape=(b, h_g_dim))
                c_h = (h, h)
                c_h = (c_h[0], c_h[1] + target_set_embeddings)
                for i in range(K):
                    attentional_softmax = tf.expand_dims(attentional_softmax, axis=2)
                    attented_features = support_set_embeddings * attentional_softmax
                    attented_features_summed = tf.reduce_sum(attented_features, axis=1)
                    c_h = (c_h[0], c_h[1] + attented_features_summed)
                    x, h_c = fw_lstm_cells_encoder(inputs=target_set_embeddings, state=c_h)
                    attentional_softmax = tf.layers.dense(x, units=k, activation=tf.nn.softmax, reuse=self.reuse)
                    self.reuse = True

            outputs = x
            print("out shape", tf.stack(outputs, axis=0).get_shape().as_list())
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            print(self.variables)
            return outputs
        '''
        if K is None:
            K = self.k
        k, b, h_g_dim = support_set_embeddings.shape
        b, h_f_dim = target_set_embeddings.shape
        attentional_softmax = torch.ones((k, b)) * 1. / k
        c_h = self.hidden
        c_h = (c_h[0], c_h[1] + target_set_embeddings)
        for i in range(K):
            attentional_softmax = attentional_softmax.unsqueeze(2).cuda()
            attented_features = support_set_embeddings * attentional_softmax
            attented_features_summed = torch.sum(attented_features, dim=0)
            c_h = (c_h[0], c_h[1] + attented_features_summed)
            output, self.hidden = self.lstm(target_set_embeddings.unsqueeze(1), c_h[1])
            attentional_softmax = self.attentional_softmax(output)
        return output


class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, batch_size=32, num_channels=1, learning_rate=1e-3, fce=False, num_classes_per_set=20,
                 num_samples_per_class=1, image_size=28, use_cuda=True):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.acc = True
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.g = Classifier(num_channels=num_channels)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()

    def train(self, mode=True):
        super().train(mode)
        self.acc = True

    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, i, :, :])
            encoded_images.append(gen_encode.cpu())

        # produce embeddings for target images
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode.cpu())
        output = torch.stack(encoded_images).cuda()

        # use fce?
        # if self.fce:
        #     outputs = self.lstm(output)

        # get similarities between support set embeddings and target
        similarites = self.dn(support_set=output[:-1], input_image=output[-1])

        # produce predictions for target probabilities
        preds = self.classify(similarites, support_set_y=support_set_y_one_hot)

        # calculate the accuracy
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())
        if self.acc:
            return accuracy, crossentropy_loss
        return preds
