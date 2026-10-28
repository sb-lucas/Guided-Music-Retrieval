from models.base_method import BaseMethod
from models.baseline.baseline import Baseline

class BaselineMethod(BaseMethod):
    def __init__(self, loss_tau, global_dim, alpha, beta):
        self.alpha, self.beta = alpha, beta     # set before super().__init__(), which calls build_model()
        super().__init__(loss_tau, global_dim)

    def build_model(self):
        return Baseline(self.alpha, self.beta)
