from sklearn import decomposition

pca = decomposition.PCA(n_components = 3)  # n_components = N; n_components should be <= no. of features
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


## plot PCA-transformed data
principalDf = pd.DataFrame(data = X_train_pca
             , columns = ['PC 1', 'PC 2', 'PC 3'])
finalDf = pd.concat([principalDf, y_train], axis = 1)


finalDf.dtypes
finalDf.head(2)


# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component PCA', fontsize = 20)

#targets = ['0', '1']
#colors = ['r', 'b']
#for target, color in zip(targets,colors):
#    indicesToKeep = finalDf['class2'] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'PC 1']
#               , finalDf.loc[indicesToKeep, 'PC 2']
#               , c = color
#               , s = 50)
#ax.legend(targets)
#ax.grid()


plt.scatter(finalDf['PC 1'], finalDf['PC 2'], alpha=0.2, c=finalDf['class2'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



