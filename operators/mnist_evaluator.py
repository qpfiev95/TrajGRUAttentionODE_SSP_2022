try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import logging
import os
from collections import namedtuple
from configuration.config import cfg
from operators.loss_funcs import *


def get_correlation(prediction, truth):
    """
    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray
    Returns
    -------
    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    eps = 1E-12
    ret = (prediction * truth).sum(axis=(3, 4)) / (
            np.sqrt(np.square(prediction).sum(axis=(3, 4))) * np.sqrt(np.square(truth).sum(axis=(3, 4))) + eps)
    ret = ret.sum(axis=(1, 2))
    return ret


class MNISTEvaluation(object):
    def __init__(self, seq_len):
        self._seq_len = seq_len
        self.begin()

    def begin(self):
        self._mse = 0
        self._mae = 0
        self._gdl = 0
        self._ssim = 0
        self._psnr = 0
        self._datetime_dict = {}
        self._total_batch_num = 0

    def clear_all(self):
        self._mse = 0
        self._mae = 0
        self._gdl = 0
        self._ssim = 0
        self._total_batch_num = 0

    def update(self, gt, pred, mask=None, start_datetimes=None):
        batch_size = gt.shape[1]
        self._total_batch_num += batch_size
        mse = mse_loss(pred, gt)
        mae = mae_loss(pred, gt)
        ssim = ssim_loss(pred, gt)
        psnr = psnr_loss(pred, gt)
        self._mse = mse
        self._mae = mae
        self._ssim = ssim
        self._psnr = psnr

    def calculate_f1_score(self):
        pass

    def calculate_stat(self):
        mse = self._mse
        mae = self._mae
        ssim = self._ssim
        return mse, mae, ssim
    def calculate_stat_test(self):
        mse = self._mse
        mae = self._mae
        ssim = self._ssim
        psnr = self._psnr
        return mse, mae, ssim, psnr

    def print_stat_readable(self, prefix=""):
        logging.getLogger().setLevel(logging.INFO)
        logging.info("s% Total sequence Number: %d" % (prefix, self._total_batch_num))
        mse, mae, ssim = self.calculate_stat()
        logging.info("MSE: %g" % (mse.mean()))
        logging.info("MAE: %g" % (mae.mean()))
        logging.info("SSIM: %g" % (ssim.mean()))

    def save_pkl(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(path, 'wb')
        logging.info("Saving HKOEvaluation to %s" % path)
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_txt_readable(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        mse, mae, ssim = self.calculate_stat()
        f = open(path, 'w')
        logging.info("Saving readable txt of MNISTEvaluation to %s" % path)
        f.write("Total Sequence Num: %d, Out Seq Len: %d.\n"
                % (self._total_batch_num,
                   self._seq_len))
        f.write("MSE: %s\n" % str(list(mse)))
        f.write("MAE: %s\n" % str(list(mae)))
        f.write("SSIM: %s\n" % str(list(ssim)))
        f.close()

    def save(self, prefix):
        self.save_txt_readable(prefix + ".txt")
        self.save_pkl(prefix + ".pkl")






