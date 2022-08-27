import numpy as np
from leopard import Layer
import leopard.functions as F
import leopard.layers as L
from leopard import utils


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    pass


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
