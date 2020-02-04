from sklearn import decomposition

####################################################################################

def get_pca(X_train):
    pca = decomposition.PCA()  # n_components = N; should be <= #features
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    #X_test_pca = pca.transform(X_test)
    print(pca.explained_variance_ratio_)
    
    ## plot PC variances
    pca_var = pca.explained_variance_ratio_
    indices = np.argsort(pca_var)[::-1]
    
    plt.figure()
    plt.title('PCA variance')
    plt.bar(range(X_train_pca.shape[1]), pca_var[indices], color='r', align ="center")
    plt.xticks(range(X_train_pca.shape[1]), indices)
    plt.xlim([-1, X_train_pca.shape[1]])
    plt.show()
    
    # Dump components relations with features:
    print(pd.DataFrame(pca.components_, columns = X_train.columns)) #,index = ['PC-1','PC-2'])
    # https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn

    
def get_pca_2(X_train):
    ## using 2 components
    pca = decomposition.PCA(n_components = 2)  # n_components = N; should be <= #features
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    #X_test_pca = pca.transform(X_test)
    return X_train_pca

  
## plot PCA-transformed data
def plot_pca2(X_train_pca, y_train):
    principalDf = pd.DataFrame(data = X_train_pca, columns = ['PC 1', 'PC 2'])
    finalDf = pd.concat([principalDf, y_train], axis = 1)
    
    plt.scatter(finalDf['PC 1'], finalDf['PC 2'], alpha=0.5, c=finalDf['status'], cmap='viridis') #alpha=0.2,cmap='viridis'
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.show()
 
#####################################################################################


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



