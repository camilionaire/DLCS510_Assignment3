import numpy as np
import matplotlib.pyplot as plt

WIDTH = .35
labels = ['10*10', '30*10','50*10', '70*10', '90*10']
ind = np.arange(len(labels)) + 1# the x locations for the groups
resEpoch = [25, 23, 18, 22, 24]
vggEpoch = [25, 24, 23, 19, 25]
# resnet = [0.8073, 0.8955, 0.9063, 0.9262, 0.9280]
# vgg = [0.8348, 0.8876, 0.9096, 0.9186, 0.9297]
plt.xticks(ind, labels)
# plt.ylim(bottom=0.6)
plt.ylabel('Accuracy')
plt.xlabel('Training Size')
# plt.grid(color='#95a5a6', axis='y')
plt.plot(ind, resEpoch, color='r', marker='o')
plt.plot(ind, vggEpoch, color='b', marker='o')
# plt.bar(ind - WIDTH / 2, resnet, color='r', width=WIDTH)
# plt.bar(ind + WIDTH / 2, vgg, color='b', width=WIDTH)
plt.legend(labels=['ResNet', 'VGG-19'])
plt.show()