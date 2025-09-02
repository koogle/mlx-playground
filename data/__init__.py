from .cifar10 import load_cifar10, CIFAR10DataLoader
from .imagenet64 import load_imagenet64, ImageNet64DataLoader

__all__ = [
    'load_cifar10', 'CIFAR10DataLoader',
    'load_imagenet64', 'ImageNet64DataLoader'
]