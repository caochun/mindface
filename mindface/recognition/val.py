"""
evaluation of lfw, calfw, cfp_fp, agedb_30, cplfw.
"""
import datetime
import os
import pickle
import argparse
from io import BytesIO
# import mxnet as mx
# import moxing as mox

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import interpolate
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from models import iresnet100, iresnet50, get_mbf, vit_t, vit_s, vit_b, vit_l, PartialFC
from runner import Network

class LFold:
    """
    LFold
    """
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        """
        split
        """
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    """
    calculate_roc
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    """calculate_acc
    """
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    """
    calculate_val
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    """calculate_val_far
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """evaluate
    """
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def load_bin(path, image_size):
    """load evalset of .bin
    """
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)
    except UnicodeDecodeError as _:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')
    data_list = []
    for _ in [0, 1]:
        data = np.zeros(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        bin_set = bins[idx]
        img = plt.imread(BytesIO(bin_set), "jpg")
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][idx][:] = img
    return data_list, issame_list


def test(data_set, backbone, batch_size, nfolds=10):
    """
    test
    """
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for data in data_list:
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            b_data = data[bb - batch_size: bb]

            time0 = datetime.datetime.now()
            img = ((b_data / 255) - 0.5) / 0.5
            net_out = backbone(ms.Tensor(img, ms.float32))
            _embeddings = net_out.asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    xnorm = 0.0
    xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            em = embed[i]
            norm = np.linalg.norm(em)
            xnorm += norm
            xnorm_cnt += 1
    xnorm /= xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, acc, _, _, _ = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc1 = np.mean(acc)
    std1 = np.std(acc)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, _, _, _ = evaluate(
        embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, xnorm, embeddings_list

# pylint: disable=C0103
def ObsToEnv(obs_data_url, data_dir):
    """
    Copy single dataset from obs to inference image.
    """
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print(f"Successfully Download {obs_data_url} to {data_dir}")
    # pylint: disable=W0703
    except Exception as e:
        print(f"moxing download {obs_data_url} to {data_dir} failed: {e}")

# pylint: disable=C0103
def ObsUrlToEnv(obs_ckpt_url, ckpt_url):
    """
    ObsUrlToEnv
    """
    try:
        mox.file.copy(obs_ckpt_url, ckpt_url)
        print(f"Successfully Download {obs_ckpt_url} to {ckpt_url}")
    # pylint: disable=W0703
    except Exception as e:
        print(f"moxing download {obs_ckpt_url} to {ckpt_url} failed: {e}")

# pylint: disable=C0103
def EnvToObs(train_dir, obs_train_url):
    """
    EnvToObs.
    """
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print(f"Successfully Download {train_dir} to {obs_train_url}")
    # pylint: disable=W0703
    except Exception as e:
        print(f"moxing upload {train_dir} to {obs_train_url} failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do verification')

    parser.add_argument('--data_url', type=str, default= '/cache/data/',
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_url', help='model to save/load',
                        default=  '/cache/checkpoint.ckpt')
    parser.add_argument('--result_url', help='result folder to save/load',
                        default= '/cache/result/')
    parser.add_argument('--device_target', type=str, default="Ascend",
                        choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--model', default='iresnet50', help='model names')
    parser.add_argument('--target', default='lfw,cfp_fp,agedb_30', help='test targets.')
    parser.add_argument('--batch-size', default=64, type=int, help='')
    parser.add_argument('--num_features', default=512, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')

    args = parser.parse_args()

    print(args)

    data_dir = args.data_url
    result_dir = '.'
    ckpt_url = args.ckpt_url

    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id, mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    image_size = [112, 112]
    time0 = datetime.datetime.now()

    if args.model == 'iresnet50':
        model = iresnet50(num_features=args.num_features)
        print("Finish loading iresnet50")
    elif args.model == 'iresnet100':
        model = iresnet100(num_features=args.num_features)
        print("Finish loading iresnet100")
    elif args.model == 'mobilefacenet':
        model = get_mbf(num_features=args.num_features)
        print("Finish loading mobilefacenet")
    elif args.model == 'vit_t':
        model = vit_t(num_features=args.num_features)
        print("Finish loading vit_t")
    elif args.model == 'vit_s':
        model = vit_s(num_features=args.num_features)
        print("Finish loading vit_s")
    elif args.model == 'vit_b':
        model = vit_b(num_features=args.num_features)
        print("Finish loading vit_b")
    elif args.model == 'vit_l':
        model = vit_l(num_features=args.num_features)
        print("Finish loading vit_l")
    else:
        raise NotImplementedError
    
    head = PartialFC(num_classes=10572, world_size=1)
    model = Network(model, head)
    
    param_dict = load_checkpoint(ckpt_url)
    load_param_into_net(model, param_dict)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    print(args.target.split(','))

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

    length = len(ver_list)
    for i in range(length):
        acc1, std1, acc2, std2, xnorm, _ = test(
            ver_list[i], model._backbone, args.batch_size, args.nfolds)
        print(f"[{ver_name_list[i]}]XNorm: {xnorm}")
        print(f"'[{ver_name_list[i]}]Accuracy: {acc1:1.5f}+-{std1:1.5f}")
        print(f"[{ver_name_list[i]}]Accuracy-Flip: {acc2:1.5f}+-%{std2:1.5f}")

    filename = 'result.txt'
    file_path = os.path.join(result_dir, filename)
    with open(file_path, 'a+') as file:
        file.write(f"[{ver_name_list[i]}]XNorm: {xnorm}")
        file.write(f"'[{ver_name_list[i]}]Accuracy: {acc1:1.5f}+-{std1:1.5f}")
        file.write(f"[{ver_name_list[i]}]Accuracy-Flip: {acc2:1.5f}+-%{std2:1.5f}")
