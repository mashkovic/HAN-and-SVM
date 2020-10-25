import time
import pandas as pd
import numpy as np
import os
from os.path import isfile, join, abspath, exists, isdir, expanduser
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
# from torchvision.models.inception import *
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from torch.optim import lr_scheduler

import random

from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


def extract_features(data, model, col):
    '''
        Code Snippet from
        https://www.kaggle.com/gennadylaptev/feature-extraction-with-pytorch-pretrained-models
    '''
    model.eval() 
    data_generator = DataLoader(data)

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        features = None
        for iter, (x, label) in enumerate(data_generator):
            output = model(x)
            if features is not None:
                features = torch.cat((features, output), 0)
            else:
                features = output
        
        features = features.view(features.size(0), -1)
        feat_df = pd.DataFrame(features.cpu().numpy(), columns=[f'model_feat_{n}' for n in range(features.size(-1))])
               
        feat_df = feat_df.fillna(0)
    
    return feat_df
