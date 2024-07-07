import os

import torch

from softs.models import SOFTS
from softs.utils.tools import get_logger


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SOFTS': SOFTS,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            get_logger().debug("Using GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device('cpu')
            get_logger().debug("Using CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
