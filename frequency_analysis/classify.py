import cv2
import numpy as np
from libs.commons import get_frequencies
from libs.FrequencySolver import FrequencySolver
import os
import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata

solver_object = FrequencySolver(num_iter=600, features=300)
solver_object(reals_path='/home/anaradutoiu/Documents/Sem_2/EAI/projects/DeepFakeDetection/faceforensics/real/c23/jpg-nonresized', fakes_path='/home/anaradutoiu/Documents/Sem_2/EAI/projects/DeepFakeDetection/faceforensics/deepfakes/c23/jpg-nonresized')
solver_object.train()
solver_object.visualize()