import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace as st

zero_thres = 1e-6

grad_img = np.load('work_dirs/erf/drn_adv.npy')
# plt.imsave('work_dirs/erf/psp_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')
plt.imsave('work_dirs/erf/drn_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/psp_adv.npy')
# plt.imsave('work_dirs/erf/psp_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')
plt.imsave('work_dirs/erf/psp_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/psa_adv.npy')
plt.imsave('work_dirs/erf/psa_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/deeplabv3_adv.npy')
plt.imsave('work_dirs/erf/deeplabv3_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/segformer_adv_whole.npy')
plt.imsave('work_dirs/erf/segformer_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')


grad_img = np.load('work_dirs/erf/deeplabv3.npy')
plt.imsave('work_dirs/erf/deeplabv3.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/psa.npy')
plt.imsave('work_dirs/erf/psa.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/segformer.npy')
plt.imsave('work_dirs/erf/segformer.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/psp.npy')
plt.imsave('work_dirs/erf/psp.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')

grad_img = np.load('work_dirs/erf/test.npy')
# plt.imsave('work_dirs/erf/psp_adv.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')
plt.imsave('work_dirs/erf/test.png', np.clip(np.clip(np.abs(grad_img)-zero_thres, 0,1e8)*1e5,0,1).mean(-1), cmap='jet')