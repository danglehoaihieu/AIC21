from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
import random
import time

# data_dir = './MNIST_data' # path to your data folder
# mnist = fetch_mldata('MNIST original', data_home=data_dir)
# print("Shape of minst data:", mnist.data.shape)

# # np.random.seed(18)
# K = 5 # 3 clusters
# N = 300


class  KMeans_clusters:
	def __init__(self, X, n_clusters=3):
		self.X = X # a array numpy NxM
		self.N, self.M = self.X.shape[:2]
		self.K = n_clusters
		la1 = np.arange(self.K)
		la2 = np.random.randint(0,self.K, (self.N-self.K,))
		self.label = np.concatenate((la1,la2))
		self.y_One_Hot = np.zeros((self.N, self.K), dtype='int')
		self.y_One_Hot[np.arange(self.N), self.label] = 1
		self.centroid = np.zeros((self.K, self.M))

	def update_centroid(self):
		a = np.zeros((self.N, self.K, self.M))
		a[np.where(self.y_One_Hot == 1)] = self.X
		count = np.sum(self.y_One_Hot, axis=0).reshape((self.K,1))
		s = np.sum(a,axis=0)
		index = np.where(count==0)
		count[np.where(count==0)] = 1
		s[index[0]] = self.centroid[index[0]]*0
		self.centroid =s/count

	def update_y_OneHot(self):
		b = np.zeros((self.N, self.K))
		b = np.argmin(cdist(self.X, self.centroid, 'euclidean'), axis=1)
		self.y_One_Hot = np.zeros((self.N, self.K), dtype='int')
		self.y_One_Hot[np.arange(self.N),b] = 1
		return b

	def has_converged(self, old, new):
		return np.array_equal(old, new)

	def  predict(self):
		loop = 0
		while True:
			self.update_centroid()
			label_new = self.update_y_OneHot()
			if  self.has_converged(self.label, label_new):
				break
			loop +=1
			self.label = label_new
		return self.centroid, self.label

	def visualize3D(self):
		color = ['r','g','b','c','m','k'][:self.K]
		marker = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd', '.',','][:self.K]
		fig = plt.figure()
		ax = plt.axes(projection ="3d")
		for i in range(self.K):
			ax.scatter3D(self.X[np.where(self.label==i),0],
						self.X[np.where(self.label==i),1],
						self.X[np.where(self.label == i),2],
						color=color[i],marker=marker[i],alpha=.8, zorder=1,linewidth=.8)

			ax.scatter3D(self.centroid[i,0],self.centroid[i,1],self.centroid[i,2],
						color='y', marker=marker[i],linewidth=10,s=100,zorder=2)
		plt.show()

	def visualize2D(self):
		color = ['r','g','b','c','m','k'][:self.K]
		marker = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd', '.',','][:self.K]
		for i in range(self.K):
			plt.plot(self.X[np.where(self.label == i),0],
					self.X[np.where(self.label == i),1],
					color=color[i],marker=marker[i],markersize=5,alpha=.8, zorder=1,linewidth=1)
			plt.scatter(self.centroid[i,0],self.centroid[i,1],
					color='y', marker=marker[i],linewidth=2,s=200, zorder=2)
		plt.show()

	def show_result(self):
		if self.M == 2: self.visualize2D()
		else: self.visualize3D()

if __name__ == '__main__':
	# np.random.seed(18)
	K = 5 # 5 clusters
	N = 100
	means = [[1, 4, 4], [5, 1, 4], [2, 7, 2], [1, 4, 5], [0, 3, 1]]
	cov = np.eye(3)

	X0 = np.random.multivariate_normal(means[0], cov, N)
	X1 = np.random.multivariate_normal(means[1], cov, N)
	X2 = np.random.multivariate_normal(means[2], cov, N)
	X = np.concatenate((X0, X1, X2), axis = 0)
	random.shuffle(X)

	model = KMeans_clusters(X, K)
	start = time.perf_counter()
	centroid, label = model.predict()
	print('Time: {} sec'.format(np.round(time.perf_counter()-start, 5)))
	print(centroid)
	model.show_result()
	