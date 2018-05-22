from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 6)
# X is the df of features, Y is the target variable value df
knn.fit(X, Y)  # the input features should be continuous & not categorical

prediction = knn.predict(X_new)
print('Prediction {}’.format(prediction))
