################################################################################
# Spatial Transformation file
#
# from :
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html#
################################################################################


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

plt.ion()

print("Hello pytorch tutorial")

from six.moves import urllib
