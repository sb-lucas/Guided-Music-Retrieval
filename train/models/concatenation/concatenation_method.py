from models.base_method import BaseMethod
from models.concatenation.concatenation import Concatenation

class ConcatenationMethod(BaseMethod):
    def build_model(self):
        return Concatenation(input_dim=self.global_dim*2, output_dim=self.global_dim)
