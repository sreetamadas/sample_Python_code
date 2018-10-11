#### steps ####

# get data


#### data pre-processing  ####
# 1. clean data - take care of missing values

# 2. check data types

# 3. create new features

# 4. subset only required colunms


# 5. convert factor colunms
df[['class','class2']] = df[['class','class2']].astype('str') 
df[['class','class2']] = df[['class','class2']].astype('category')


# 6. create train-test data : 2 class
X = df[['X1','X2','X3','X4','X6']]  
y = df[['class2']]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# use "stratify" option for consistent class distribution between training and test sets
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65


# 7. normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_train_scaled.shape


# 8. use PCA, instead of actual features, if too many features (or, do a feature importance map)
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print((pca.explained_variance_ratio_)*100)



### model building ###

# basic model building


# cross-validation & hyper-parameter tuning


# combine different models


# bagging data sets








