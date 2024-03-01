import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, IncrementalPCA

# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x = mnist.data
y = mnist.target

# %%
x.head()

# %%
# scaling the features
X_scaled = scale(x)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)

# %%
pca = PCA(svd_solver='randomized', random_state=42)
pca.fit(X_train)

# %%
#Making the screeplot - plotting the cumulative variance against the number of components
%matplotlib inline
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# %%
pca = IncrementalPCA(n_components=400)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test) 

# %%
X_train.shape

# %%
# model
model = SVC(C=10, gamma = 0.001, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

# %%
def report(model):
    print(model.get_params())
    preds=model.predict(X_test)
    print(metrics.accuracy_score(y_test,preds)) 
    cm=metrics.confusion_matrix(y_test,preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
    display_labels = model.classes_)
    cm_display.plot()        
    plt.show()
    print(metrics.classification_report(y_test,preds))


# %%
report(model)

# %%
from sklearn import tree

Dt = tree.DecisionTreeClassifier(random_state=0)
Dt.fit(X_train, y_train) 

# %%
report(Dt)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

# %%
report(knn)

# %%
from sklearn.linear_model import LogisticRegression

max_iter = 100 )
           
LR.fit(X_train, y_train)
report(LR)

LR = LogisticRegression(penalty = "l2", multi_class="multinomial",

