from PIL import Image, ImageDraw
import random
import os
import cv2
import numpy as np
import torch
import sys
import os

mo = torch.nn.ConvTranspose2d(3, 3, 4, 2, 1)
x = torch.ones((1, 3, 64, 64))
y = mo(x)
print(y.shape)
