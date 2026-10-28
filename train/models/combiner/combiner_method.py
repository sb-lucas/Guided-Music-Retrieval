from models.base_method import BaseMethod
from models.combiner.combiner import Combiner

class CombinerMethod(BaseMethod):
    def build_model(self):
        return Combiner(clap_feature_dim=self.global_dim, projection_dim=256, hidden_dim=128)
