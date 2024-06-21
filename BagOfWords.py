#env name 315Final
#ctrl + shft p: python: select envionrments
import torch
import pandas as pd
import numpy as np
import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm

train, test = datasets.load_dataset('imdb', split=['train', 'test'])

print(train[0])