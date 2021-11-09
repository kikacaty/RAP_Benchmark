import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace as st

grad_img = np.load('work_dirs/erf/deeplabv3.npy')
plt.imsave('work_dirs/erf/deeplabv3.png', np.clip((np.abs(grad_img))*(1/np.abs(grad_img).max())*100,0,1))

grad_img = np.load('work_dirs/erf/psa.npy')
plt.imsave('work_dirs/erf/psa.png', np.clip((np.abs(grad_img))*(1/np.abs(grad_img).max())*100,0,1))

grad_img = np.load('work_dirs/erf/segformer.npy')
plt.imsave('work_dirs/erf/segformer.png', np.clip((np.abs(grad_img))*(1/np.abs(grad_img).max())*100,0,1))