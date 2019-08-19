import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

# The digits dataset
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
# print(images_and_labels[0][0])
# plt.imshow(images_and_labels[1][0]);plt.show()
print(np.max(digits.images))

n_samples = len(digits.images)
print(n_samples)
data = digits.images.reshape((n_samples, -1))

model = KMeans(init='random', n_clusters=10, random_state=0)
model.fit(data)

print('Model evaluation with random  centroids initialization: ')
print(metrics.homogeneity_completeness_v_measure(digits.target, model.labels_),
      metrics.mutual_info_score(digits.target, model.labels_))

model = KMeans(init='k-means++', n_clusters=10, random_state=0)
model.fit(data)

print('Model evaluation with kmeans++ centroids initialization: ')
print(metrics.homogeneity_completeness_v_measure(digits.target, model.labels_),
      metrics.mutual_info_score(digits.target, model.labels_))

model = KMeans(init='k-means++', n_clusters=10, random_state=0)
_data = data/16.0
model.fit(_data)

print('Model evaluation with data normalisation: ')
print(metrics.homogeneity_completeness_v_measure(digits.target, model.labels_),
      metrics.mutual_info_score(digits.target, model.labels_))

pca = PCA(n_components=10).fit(data)
model = KMeans(init=pca.components_, n_clusters=10, random_state=0)
model.fit(data)


print('Model evaluation with data dimension reduction: ')
print(metrics.homogeneity_completeness_v_measure(digits.target, model.labels_),
      metrics.mutual_info_score(digits.target, model.labels_))

cluster_results = np.hstack((data, model.labels_.reshape(-1, 1)))

# get certain class
unique_classes = np.unique(model.labels_)
selected_class = unique_classes[5]
print(selected_class)

selected_class_data = np.array([_res[:-1] for _res in cluster_results if _res[-1] == selected_class])

# N = 25
# fig = plt.figure()
# for i, _img in enumerate(selected_class_data[:N]):
#     ax = fig.add_subplot(np.sqrt(N), np.sqrt(N), i+1)
#     ax.imshow(_img.reshape(8, 8))
#     ax.axis('off')
# fig.show()

N = 100
fig = plt.figure(figsize=(16, 16))
for _cls in range(10):
    selected_class_data = np.array([_res[:-1] for _res in cluster_results if _res[-1] == _cls])
    for i, _img in enumerate(selected_class_data[:10]):
        ax = fig.add_subplot(np.sqrt(N), np.sqrt(N), _cls*10+i+1)
        # ax.imshow(_img.reshape(8, 8), cmap='gray')
        ax.imshow(_img.reshape(8, 8))
        ax.axis('off')
fig.show()
fig.savefig('mnist_cluster_res.png', dpi=1024)


