import os
import weakref

import numpy as np
from leopard.core import Parameter
import leopard.functions as F


class Layer:
    def __init__(self):
        self._params = set()  # 変数名を複数形に

    def __setattr__(self, __name: str, __value: Parameter) -> None:
        if isinstance(__value, (Parameter, Layer)):  # Layerも追加する
            self._params.add(__name)
        super().__setattr__(__name, __value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()  # Layerからパラメータを取り出す
            else:
                yield obj  # yieldで処理を一旦停止して値を返す

    def cleargrads(self):
        # Layerが持つ全てのパラメータの勾配をリセット
        for param in self.params():
            param.cleargrad()


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:  # in_sizeが指定されていない場合は後回し
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # データを流すタイミングで重みを初期化
        if self.W.data is None:
            self.in_size = x.shape[1]
            # xp = cuda.get_array_module(x)
            self._init_W(x)

        y = F.linear(x, self.W, self.b)
        return y
