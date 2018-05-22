from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 6)

# X is the df of features, Y is the target variable value df
y = df['party'].values
X = df.drop('party', axis=1).values

# fit model on training data
knn.fit(X, Y)  # the input features should be continuous & not categorical

# predict on validation/ test data
prediction = knn.predict(X_new)
print('Prediction {}’.format(prediction))
