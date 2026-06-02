import torch.nn as nn

def get_activation(active_name='relu'):
    """Selecciona la activación de forma segura y eficiente."""
    # Convertimos a minúsculas para evitar errores si escribes 'ReLU' o 'relu'
    name = active_name.lower() if isinstance(active_name, str) else 'identity'
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leakyrelu': nn.LeakyReLU(inplace=True),
        'prelu': nn.PReLU(),
        'rrelu': nn.RReLU(inplace=True),
        'relu6': nn.ReLU6(inplace=True),
        'elu': nn.ELU(inplace=True),
        'selu': nn.SELU(inplace=True),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softplus': nn.Softplus(),
        'identity': nn.Identity()
    }
    # Si el nombre no existe, por defecto usamos ReLU (evita que el código muera)
    return activations.get(name, nn.ReLU(inplace=True))

def get_pooling(pool_name='maxpool2d', kernel_size=2):
    """Pooling sin usar eval()."""
    name = pool_name.lower() if isinstance(pool_name, str) else 'maxpool2d'
    pools = {
        'maxpool1d': nn.MaxPool1d(kernel_size=kernel_size),
        'maxpool2d': nn.MaxPool2d(kernel_size=kernel_size),
        'maxpool3d': nn.MaxPool3d(kernel_size=kernel_size),
        'avgpool1d': nn.AvgPool1d(kernel_size=kernel_size),
        'avgpool2d': nn.AvgPool2d(kernel_size=kernel_size),
        'avgpool3d': nn.AvgPool3d(kernel_size=kernel_size)
    }
    return pools.get(name, nn.MaxPool2d(kernel_size=kernel_size))

def get_dropout(dropout_name='dropout', p=0.5):
    """Dropout sin usar eval()."""
    name = dropout_name.lower() if isinstance(dropout_name, str) else 'dropout'
    # Validamos que p sea una probabilidad razonable
    p = float(p) if 0 <= float(p) <= 1 else 0.5
    dropouts = {
        'dropout': nn.Dropout(p=p),
        'dropout2d': nn.Dropout2d(p=p),
        'dropout3d': nn.Dropout3d(p=p)
    }
    return dropouts.get(name, nn.Dropout(p=p))