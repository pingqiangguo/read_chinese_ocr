from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train


def get_network(name):
    """Get a network by name.

    """
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
            return VGGnet_test()
        elif name.split("_")[1] == "train":
            return VGGnet_train()
        else:
            raise Exception('get_network can only be used to load VGG_test and VGG_train model: {}'.format(name))
    else:
        raise Exception('get_network can only be used to load VGG_test and VGG_train model: {}'.format(name))
