# -*- coding: utf-8 -*-
# """
# tensorflow_model_utils.py
# Created on Dec 29, 2024
# @ Author: Mazhar
# """

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import tensorflow as tf
from IPython.display import Markdown, display
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split

#compute accuracy and percent error 
def accu(y_pred,y_test):
    accuracy = (1- np.mean(abs(y_pred-y_test)/y_test))*100
    PE = np.mean(abs(y_pred-y_test)/y_test)*100
    return accuracy, PE