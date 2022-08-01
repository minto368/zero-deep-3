# =============================================================================
# step23.pyからstep32.pyまではsimple_coreを利用
is_simple_core = True # False
# =============================================================================

if is_simple_core:
    from leopard.core_simple import Variable
    from leopard.core_simple import Function
    from leopard.core_simple import using_config
    from leopard.core_simple import no_grad
    from leopard.core_simple import as_array
    from leopard.core_simple import as_variable
    from leopard.core_simple import setup_variable

else:
    from leopard.core import Variable
    from leopard.core import Parameter
    from leopard.core import Function
    from leopard.core import using_config
    from leopard.core import no_grad
    from leopard.core import test_mode
    from leopard.core import as_array
    from leopard.core import as_variable
    from leopard.core import setup_variable
    from leopard.core import Config
    from leopard.layers import Layer
    from leopard.models import Model
    from leopard.datasets import Dataset
    from leopard.dataloaders import DataLoader
    from leopard.dataloaders import SeqDataLoader

    import leopard.datasets
    import leopard.dataloaders
    import leopard.optimizers
    import leopard.functions
    import leopard.functions_conv
    import leopard.layers
    import leopard.utils
    import leopard.cuda
    import leopard.transforms

setup_variable()
__version__ = '0.0.13'