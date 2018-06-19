import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns;sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
X , y = make_blobs(n_samples=50 , centers=2 , random_state=0 , cluster_std=0.60)


model = SVC(kernel='linear')
model.fit(X , y)
w = model.coef_
b = model.intercept_
plt.scatter(X[: , 0] , X[: , 1] , c = y , s = 50 , cmap = 'autumn')
x = np.linspace(0, 7)
print(w.shape , x.shape)
y = (-float(b) - w[0, 0] * x) / w[0, 1]
plt.plot(x , y)
plt.show()