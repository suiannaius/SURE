import numpy as np
import sklearn.metrics as metrics
import pymia.evaluation.metric as m


def ece_binary(probabilities, target, n_bins=10, threshold_range: tuple = None, mask=None, out_bins: dict = None,
               bin_weighting='proportion'):

    n_dim = target.ndim

    bin_accuracy, bin_confidence, bin_count, non_zero_bins = \
        binary_calibration(probabilities, target, n_bins, threshold_range, mask)

    bin_proportions = _get_proportion(bin_weighting, bin_count, non_zero_bins, n_dim)

    if out_bins is not None:
        out_bins['bins_count'] = bin_count
        out_bins['bins_avg_confidence'] = bin_confidence
        out_bins['bins_positive_fraction'] = bin_accuracy
        out_bins['bins_non_zero'] = non_zero_bins

    ece = (np.abs(bin_confidence - bin_accuracy) * bin_proportions).sum()
    return ece


def binary_calibration(probabilities, target, n_bins=10, threshold_range: tuple = None, mask=None):
    """
    Input:
    针对每一类别c:
    probabilities: [NHWD, 1]
    target: [NHWD, 1]
    """
    assert probabilities.ndim == target.ndim, f"The ndim of probabilities and target should be the same, but got {probabilities.ndim} and {target.ndim}."
    if mask is not None:
        probabilities = probabilities[mask]
        target = target[mask]

    if threshold_range is not None:
        low_thres, up_thres = threshold_range
        mask = np.logical_and(probabilities < up_thres, probabilities > low_thres)
        probabilities = probabilities[mask]
        target = target[mask]

    bin_accuracy, bin_confidence, bin_count, non_zero_bins = \
        _binary_calibration(target.flatten(), probabilities.flatten(), n_bins)

    return bin_accuracy, bin_confidence, bin_count, non_zero_bins


def _binary_calibration(target, pred_positive, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1) # bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-8]
    binids = np.digitize(pred_positive, bins) - 1 # 返回每个pred_positive值对应的区间索引

    bin_pred_sums = np.bincount(binids, weights=pred_positive, minlength=n_bins) # 返回每个 bin 区间的概率值之和, dim: [n_bins, ]
    bin_true_sums = np.bincount(binids, weights=target, minlength=n_bins) # 计算每个区间内真实正类的数量, dim: [n_bins, ]
    bin_total = np.bincount(binids, minlength=n_bins) # 计算每个区间内的样本总数, dim: [n_bins, ]

    nonzero = bin_total != 0
    bin_count = bin_total[nonzero] # 每个非零区间内的样本总数
    bin_accuracy = (bin_true_sums[nonzero] / bin_total[nonzero]) # 每个非零区间的acc
    bin_confidence = (bin_pred_sums[nonzero] / bin_total[nonzero]) # 每个非零区间的confidence
    
    # print('\nbin_pred_sums:\n', bin_pred_sums, '\n')
    # print('bin_true_sums:\n', bin_true_sums, '\n')
    # print('bin_total:\n', bin_total, '\n')
    # print('nonzero:\n', nonzero, '\n')
    # print('bin_count:\n', bin_count, '\n')
    # print('bin_accuracy:\n', bin_accuracy, '\n')
    # print('bin_confidence:\n', bin_confidence, '\n')
    return bin_accuracy, bin_confidence, bin_count, nonzero


def _get_proportion(bin_weighting: str, bin_count: np.ndarray, non_zero_bins: np.ndarray, n_dim: int):
    if bin_weighting == 'proportion':
        bin_proportions = bin_count / bin_count.sum()
    elif bin_weighting == 'log_proportion':
        bin_proportions = np.log(bin_count) / np.log(bin_count).sum()
    elif bin_weighting == 'power_proportion':
        bin_proportions = bin_count**(1/n_dim) / (bin_count**(1/n_dim)).sum()
    elif bin_weighting == 'mean_proportion':
        bin_proportions = 1 / non_zero_bins.sum()
    else:
        raise ValueError('unknown bin weighting "{}"'.format(bin_weighting))
    return bin_proportions


def uncertainty(prediction, target, thresholded_uncertainty, mask=None):
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
        thresholded_uncertainty = thresholded_uncertainty[mask]

    tps = np.logical_and(target, prediction)
    tns = np.logical_and(~target, ~prediction)
    fps = np.logical_and(~target, prediction)
    fns = np.logical_and(target, ~prediction)

    tpu = np.logical_and(tps, thresholded_uncertainty).sum()
    tnu = np.logical_and(tns, thresholded_uncertainty).sum()
    fpu = np.logical_and(fps, thresholded_uncertainty).sum()
    fnu = np.logical_and(fns, thresholded_uncertainty).sum()

    tp = tps.sum()
    tn = tns.sum()
    fp = fps.sum()
    fn = fns.sum()

    return tp, tn, fp, fn, tpu, tnu, fpu, fnu


def error_dice(fp, fn, tpu, tnu, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fn + fp + fnu + fpu + tnu + tpu) == 0):
        return 1.
    return (2 * (fnu + fpu)) / (fn + fp + fnu + fpu + tnu + tpu)


def error_recall(fp, fn, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fn + fp) == 0):
        return 1.
    return (fnu + fpu) / (fn + fp)


def error_precision(tpu, tnu, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fnu + fpu + tpu + tnu) == 0):
        return 1.
    return (fnu + fpu) / (fnu + fpu + tpu + tnu)


def dice(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    d = m.DiceCoefficient()
    d.confusion_matrix = m.ConfusionMatrix(prediction, target)
    return d.calculate()


def confusion_matrx(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    cm = m.ConfusionMatrix(prediction, target)
    return cm.tp, cm.tn, cm.fp, cm.fn, cm.n


def accuracy(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    a = m.Accuracy()
    a.confusion_matrix = m.ConfusionMatrix(prediction, target)
    return a.calculate()


def log_loss_sklearn(probabilities, target, labels=None):
    _check_ndarray(probabilities)
    _check_ndarray(target)

    if probabilities.shape[-1] != target.shape[-1]:
        probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    else:
        probabilities = probabilities.reshape(-1)
    target = target.reshape(-1)
    return metrics.log_loss(target, probabilities, labels=labels)


def entropy(p, dim=-1, keepdims=False):
    # exactly the same as scipy.stats.entropy()
    return -np.where(p > 0, p * np.log(p), [0.0]).sum(axis=dim, keepdims=keepdims)


def _check_ndarray(obj):
    if not isinstance(obj, np.ndarray):
        raise ValueError("object of type '{}' must be '{}'".format(type(obj).__name__, np.ndarray.__name__))