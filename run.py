from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from model import model

dataset = pd.read_csv("fashion-mnist_train.csv", sep = ",", header = 0)
testset = pd.read_csv("fashion-mnist_test.csv", sep = ",", header = 0)
X = dataset.iloc[:, 1:]
Y = dataset.iloc[:, 0]
X_testval = testset.values
m = X.shape[0]

X, X_cv, Y, Y_cv = train_test_split(X.values, Y.values, test_size = 0.3)
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255
X_cv = X_cv.reshape(-1, 28, 28, 1).astype('float32') / 255
Y = Y.astype('int')
Y_cv = Y_cv.astype('int')

features = model(X, Y, X_cv, Y_cv, learning_rate = 0.0001, num_epochs = 10, minibatch_size = 64, drop_conv = 0.8, alpha = 0.6, features = 32, print_cost = True)

tsne_obj = TSNE(n_components=2,
                         init='pca',
                         random_state=101,
                         method='barnes_hut',
                         n_iter=500,
                         verbose=2)
tsne_features = tsne_obj.fit_transform(features)

obj_categories = ['T-shirt/top','Trouser','Pullover','Dress',
                  'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'
                 ]
colors = plt.cm.rainbow(np.linspace(0, 1, 10))
plt.figure(figsize=(10, 10))

for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):
    plt.scatter(tsne_features[np.where(Y == c_group), 0],
                tsne_features[np.where(Y == c_group), 1],
                marker='o',
                color=c_color,
                linewidth='1',
                alpha=0.8,
                label=c_label)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE on Testing Samples')
plt.legend(loc='best')
plt.savefig('clothes-dist.png')
plt.show(block=False)
