import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50
from collections import defaultdict

from configs import bcolors
from utils import (init_coverage_tables, neuron_to_cover, neuron_covered,
                   update_coverage, diverged, decode_label,
                   deprocess_image, compute_mean_std,
                   constraint_light, normalize)