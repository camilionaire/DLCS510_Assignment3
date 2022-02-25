# import numpy as np
# import matplotlib.pyplot as plt
# data = [[30, 25, 50, 20],
# [40, 23, 51, 17],
# [35, 22, 45, 19]]
# X = np.arange(4)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

# plt.show()

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# langs = ['C', 'C++', 'Java', 'Python', 'PHP']
# students = [23,17,35,29,12]
# ax.bar(langs, students, bottom = 0, color = 'r')
# # ax.set(plt.gca, 'xticklabel', langs)
# # ax.set_xticks(langs)
# plt.show()

import numpy as np
# import pandas as pd
# from pandas import Series, DataFrame
import matplotlib.pyplot as plt
WIDTH = .35
labels = ['10*10', '30*10','50*10', '70*10', '90*10']
ind = np.arange(len(labels)) + 1# the x locations for the groups
resnet = [0.8073, 0.8955, 0.9063, 0.9262, 0.9280]
vgg = [0.8348, 0.8876, 0.9096, 0.9186, 0.9297]
plt.xticks(ind, labels)
plt.ylim(bottom=0.6)
# plt.yticks(np.arange(.6, 1.0, .1))
plt.bar(ind - WIDTH / 2, resnet, color='r', width=WIDTH)
plt.bar(ind + WIDTH / 2, vgg, color='b', width=WIDTH)
plt.legend(labels=['ResNet', 'VGG-19'])
plt.show()