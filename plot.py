# 结果可视化，仅供参考（文件名称/位置需要自己更改）
import numpy as np
import matplotlib.pyplot as plt

data = np.load('results/NLSE_params.npz')
plt.style.use('seaborn-ticks')
params = data['params']
logvar = data['logvar']
y1 = np.var(params, axis=0)
y2 = np.mean(np.exp(logvar),axis=0)

x = np.arange(params.shape[1])

fig = plt.figure(dpi=120)
ax1 = fig.add_subplot(111)
ax1.bar(x, y1, color='b')
ax2 = ax1.twinx()  # this is the important function
ax2.bar(x, -y2*10, color='r')


ax1.set_ylabel('var(μ)',color='b')
ax2.set_ylabel('Mean(σ2)', color='r')
ax1.set_ylim(-0.4,1)
ax2.set_ylim(-0.4,1)
ax2.set_xlabel('Latent Parameters')
ax1.set_yticklabels(['', '', '0.0','0.2', '0.4', '0.6', '0.8', '1.0'],color='b')
ax2.set_yticklabels(['0.04','0.02', '0.0', '', '','', '', '', '', ''],color='r')
plt.show()