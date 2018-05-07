# this is a practice problem from kaggle
# Iowa house price prediction problem (KAGGLE)

import pandas as pd 
main_file_path = '../input/train.csv' 
data = pd.read_csv(main_file_path) 
#print('hello world') 
data.head()


#data.describe()
print(data.describe())

## seeing all columns of first few rows on the terminal
get_ipython().magic(u'less ../input/train.csv')  # % less ../input/train.csv

## see list of columns in the df 
print(data.columns)


# pulling out a single column (here the Y variable of house prices) & storing as series 
y = data.SalePrice 
print(y.head())


# selecting columns of interest & obtaining details 
columns_of_interest = ['Id', 'SalePrice'] 
two_columns_of_data = data[columns_of_interest] 
two_columns_of_data.describe() 
#data['Id', 'SalePrice'].head() ## gives error - why ? correct notation -> data[['LotArea','TotRmsAbvGrd']].head()

# selecting columns of interest & obtaining details 
columns_of_interest = ['LotArea','TotRmsAbvGrd','FullBath','BedroomAbvGr','OverallQual','OverallCond','YearBuilt', 'SalePrice'] 
sel_columns_of_data = data[columns_of_interest] 
sel_columns_of_data.describe()


# choosing prediction target 
#y = data.SalePrice 

# choosing predictors 
#predictors = ['LotArea','TotRmsAbvGrd','FullBath','BedroomAbvGr','OverallQual','OverallCond','YearBuilt'] #, 'SalePrice']
predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
x = data[predictors] 



# MODEL BUILDING : decision tree
from sklearn.tree import DecisionTreeRegressor 

# step 1: Define model 
housePrice_model = DecisionTreeRegressor() 
# step 2: Fit model 
housePrice_model.fit(x, y)
# step 3: predict using model 
# step 4: validate


# check predictions on the training set 
#print("raw data") 
#print(sel_columns_of_data.head()) 
print("Making predictions for the following 5 houses (predictors x):") 
print(x.head()) 
print("The predictions (y') are") 
print(housePrice_model.predict(x.head()))



# checking error on training data: 
#this is to understand the formula, & not as a real validation of the model 
from sklearn.metrics import mean_absolute_error

predicted_home_prices = housePrice_model.predict(x)
mean_absolute_error(y, predicted_home_prices)



## performance should be measured on a separate validation set - create the validation set & test prediction accuracy 
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target 
# The split is based on a random number generator.
# the random_state argument guarantees the same split every time the script runs. 
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

# step 1: Define model 
housePrice_model = DecisionTreeRegressor() 
# step 2: Fit model 
housePrice_model.fit(train_X, train_y) 
# step 3: get predicted prices on validation data 
val_predictions = housePrice_model.predict(val_X) 
# step 4: validate 
print(mean_absolute_error(val_y, val_predictions))



# experimenting with different models by varying model parameters 
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html 

# max_leaf_nodes: 1 argument to control overfitting vs underfitting. 
# The more the leaves in the model, the more we move from underfitting to overfitting 

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val): 
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0) 
    # fit model 
    model.fit(predictors_train, targ_train) 
    # predict with model 
    preds_val = model.predict(predictors_val) 
    # calculate validation score 
    mae = mean_absolute_error(targ_val, preds_val) 
    return(mae) 

# compare MAE with differing values of max_leaf_nodes 
for max_leaf_nodes in [25, 35, 40, 50, 60, 75, 85, 100]:   #5, 50, 500, 5000    #5, 25, 50, 100, 500, 5000
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) 
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
    
## problem with decision tree: A deep tree with lots of leaves will overfit 
# (because each prediction is coming from only the few houses at its leaf). 
## A shallow tree with few leaves will perform poorly (it fails to capture many distinctions in the raw data)



## random forest model 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error 

forest_model = RandomForestRegressor()   # define
forest_model.fit(train_X, train_y)       # fit
rf_pred = forest_model.predict(val_X)    # predict

print(mean_absolute_error(val_y, rf_pred))




## checking the model on unseen test data
# Read the test data
test = pd.read_csv('../input/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictors]

# Use the model to make predictions
predicted_prices = forest_model.predict(test_X)
print(predicted_prices)




## TRY: do feature selection



## prepare submission file
# https://www.kaggle.com/dansbecker/submitting-from-a-kernel
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)

