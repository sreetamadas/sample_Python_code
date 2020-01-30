from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


# types of ML algo: 
#       logistic, SVM, tree-methods (DT), Bayesian models
#       generalised linear model/GLM (for regression?)
#       neural net/MLP, CNN & variants (deep learning based methods, transfer learning)
#       ensemble methods: bagging, random forest, boosting, stacking

#       https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205?gi=f30deb598cb4    ****
#       https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840
#       https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
#       https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/
# for a comprehensive list, see websites: machinelearningmastery, scikitlearn


###  voting classifier - weighted, unweighted
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html


# optimised SVC MODEL 
model_svc_opt = SVC(C=1000, kernel='linear')
# other SVC
model_svc = SVC()

# optimised RF
model_rf_opt = RandomForestClassifier(max_features = 3, min_samples_split = 10, n_estimators = 501) 
# other RF
model_rf = RandomForestClassifier(max_features='auto', min_samples_split=2, n_estimators = 501)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=501, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

# knn
model_knn = KNeighborsClassifier(metric='euclidean', n_neighbors=5)

## voting classifier: hard
eclf_hrd = VotingClassifier(estimators=[('svc_opt', model_svc_opt), ('svc', model_svc), 
                                        ('rf_opt', model_rf_opt), ('rf', model_rf),
                                        ('knn', model_knn)], voting='hard')

## fit model - vary ML algo & hyperparameters
eclf_hrd.fit(X_train_scaled, y_train)


## voting classifier 2 - soft voting
# optimised SVC MODEL 
model_svc_opt = SVC(C=1000, kernel='linear', probability=True)
# other SVC
model_svc = SVC(probability=True)
eclf_sft = VotingClassifier(estimators=[('svc_opt', model_svc_opt), ('svc', model_svc), 
                                        ('rf_opt', model_rf_opt), ('rf', model_rf),
                                        ('knn', model_knn)], voting='soft')



