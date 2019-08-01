import cv2
import numpy as np
import 下 as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pylab as plt
#scipy是数学包
from scipy import stats

def Gaussian_mixture_model():
    return

def simulate():
    #产生身高数据
    np.random.seed(100)
    #norm是正态分布，180均值，8标准差
    sim_data_boy = np.random.normal(180, 8, 2000)
    pd.Series(sim_data_boy).hist(bins = 200)
    sim_data_girl = np.random.normal(160, 6, 2000)
    pd.Series(sim_data_girl).hist(bins = 200)
    plt.imshow()


simulate()
