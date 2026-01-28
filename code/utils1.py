import numpy as np
import sklearn.metrics
import copy
import torch.utils.data as data
import torch
import numpy as np

class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        from scipy.optimize import linear_sum_assignment
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
        return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size


class data_loder(data.Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_data(self, img, txt, label):
        size, size1 = img.__len__(), txt.__len__()

        shuffle_ix = np.random.permutation(np.arange(size))
        img = img[shuffle_ix]
        txt = txt[shuffle_ix]
        label = label[shuffle_ix]

        assert (size == size1)
        data1, data2, data3 = [], [], []
        alldata1, alldata2, alldata3 = [], [], []
        for i in range(size):
            temp_i = i % self.batch_size
            if temp_i < self.batch_size:
                data1.append(img[i])
                data2.append(txt[i])
                data3.append(label[i])
            if data1.__len__() == self.batch_size or i == size - 1:
                d1, d2, d3 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data3)
                alldata1.append(d1)
                alldata2.append(d2)
                alldata3.append(d3)
                data1.clear()
                data2.clear()
                data3.clear()
        self.data1 = alldata1
        self.data2 = alldata2
        self.data3 = alldata3



    def __getitem__(self, index):
        img, txt, target = np.array(self.data1[index]), np.array(self.data2[index]), self.data3[index]
        img = torch.tensor(img, dtype=torch.float32)
        txt = torch.tensor(txt, dtype=torch.float32)
        target = np.array(target)-1
        target = np.squeeze(target)
        return img, txt, target