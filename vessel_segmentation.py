import tensorflow as tf
from AngioNet_model import AngioNet
import os
from PIL import Image
import os

dir = os.path.join('data', "training", "images")


model = AngioNet(L1=0, L2=0, DL_weights=None)
