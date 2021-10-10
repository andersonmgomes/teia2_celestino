import numpy as np
from sklearn.svm import SVC
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
svc_model = SVC()
svc_model.fit(X, y)

print(svc_model.predict([[-0.8, -1]]))