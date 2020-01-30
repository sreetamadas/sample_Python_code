#####  dataframe: creation &  manipulations  #####

#####################   CREATION   ########################
gender = ['male','male','male','female','female','female','squirrel']
pet1 =['dog','cat','dog','cat','dog','squirrel','dog']
pet2 =['dog','cat','cat','squirrel','dog','cat','cat']

d = pd.DataFrame(np.column_stack([gender, pet1, pet2]),columns=['gender', 'petl', 'pet2'])
d['points'] = np.where( ( (d['gender'] == 'male') & (d['pet1'] == d['pet2'] ) ) | 
                       ( (d['gender'] == 'female') & (d['pet1'].isin(['cat','dog'] ) ) ), 5, 0)


#####  rename cols  ######
oldname = raw_input("enter 1st columnname: ")  # AL1
df = df.rename(columns={oldname: 'newname'})



#######################    CHECKS    #####################

#### running checks on the input data as dataframe ####
# checking types of data in different cols 
df.dtypes
df.info()

# get summary statistics
df.describe()
df['colname'].value_counts()

# df.shape[0] counts num_rows in df; df.shape[1] counts num_cols in df
df.shape



#####   missing data handling  ####
# GOOGLE: how to detect NA values in python ;  how to remove rows with NAN values in python
# https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
# https://machinelearningmastery.com/handle-missing-data-python/

# missing data
print(df.isnull())
print(df.notnull())

# count no. of rows with missing values
print(df.isnull().sum())



#######   drop unnecessary cols (lot of missing values, or containing constant value)  ######
# method 1
df.drop('Col_not_required', axis=1, inplace=True)
# method 2
del df['col_not_reqd']
# method 3
df = df.drop(['col1', 'col2'], axis=1)

# remove columns with constants
p_col = col  # original col
cons_columns = df.nunique()==1  # print this to get which cols have constants
cons_index = np.where(cons_columns)[0]
df = df.drop(columns=df.columns[cons_index])
col = df.columns   # columns remaining after dropping constant col.
rem_col = set(p_col)-set(col)

# remove col with missing values
p_col = col
missing_per = np.sum(df.isnull()) / len(df)
missing_index = np.where(missing_per > miss_thresh)[0]
df = df.drop(columns=df.columns[missing_index])
col = df.columns
rem_col = set(p_col)-set(col)

# remove rows with missing val
missing_per_row = np.sum(df.T.isnull()) / len(df)
missing_index_row = np.where(missing_per_row  > miss_thresh)[0]
p_row = df.shape[0]
df = df.drop(df.index[missing_index_row])
rem_row = p_row - df.shape[0]



####   impute missing values  #####
# code for adding missing values (median/mean or fill using previous value or next value) 
# https://pandas.pydata.org/pandas-docs/stable/missing_data.html
df = df.fillna(method='pad', limit=3)    # using previous value (up to 3 places)
df = df.fillna(method='bfill')  # backfill: fill using next value  ; ffill= using previous value
df[selected_numeric_columns] = df[selected_numeric_columns].fillna(df[selected_numeric_columns].mean())   # impute with mean
# below imputation for categories; can also use if condition to fill specific values corresponding to value in other cols
df[categorical_cols] = df[categorical_cols].fillna("nan")  


## keep rows with finite values in a column
df_cleaned = df[np.isfinite(df['col_X'])]




######   duplicate data handling   #######
####  remove all rows with duplicate values in selected cols
df_single = df.drop_duplicates(subset=['Date', 'Shift', 'Index'], keep=False)
# keep 1st row among duplicates
df.drop_duplicates(inplace=True)  # remove rowwise duplicates, using all columns
d2 = d2.drop_duplicates(subset=['subject'], keep='first')  # using select columns
# https://stackoverflow.com/questions/23667369/drop-all-duplicate-rows-in-python-pandas


####  get all rows with duplicate values in selected cols
df_multi = pandas.concat(g for _, g in df.groupby(['Date', 'Shift', 'Index']) if len(g) > 1)


### remove columnwise duplicates
col = list(df.T.drop_duplicates().T) # Columnwise duplicates
df =  df[col]




#####  remove multi-collinearity : for features with corr > 0.9, keep one & drop rest  #########
# correlation for numeric to numeric columns (def: Pearson; others:kendall, spearman)
corr_num_num = df[num_columns].corr()

"""
def cramers_v(x, y):    # x &  y are categorical data columns
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def cat_num(cat,num,b=2):
    #cat_num : converts numerical column to categorical column by quartile binning 
    #          and performs cramer's v correlation  
    # cat - categorical column ;   num - numerical column
    num = pd.Series(num).replace(to_replace=999999,value=-1)
    bb = pd.qcut(num,b,duplicates='drop')
    return cramers_v(bb, cat)

# cramer's v correlation for categorical to categorical columns
corr_cat_cat = df[cat_columns].corr(method=cramers_v)
# cramer's v correlation for categorical to binned numerical columns
corr_cat_nu = df[cat_columns+num_columns].corr(method=cat_num)
corr_cat_nu = cat_nu.loc[num_columns,cat_columns]
"""



#####   outlier detection & removal  #####
# use of boxplots
# other methods: Angle-based Outlier Detector (ABOD), Cluster-based Local Outlier Factor, Histogram-base Outlier Detection (HBOS) 
#                Feature Bagging, Isolation Forest, K Nearest Neighbors (KNN), Local Outlier Factor (LOF),
#                Minimum Covariance Determinant (MCD),  One-class SVM (OCSVM), Principal Component Analysis (PCA)
# https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/
"""from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA as PCA_pyod"""




##################   MANIPULATIONS   #################
####  concatenation  ####
# ADD 1 df below the other
newdata = pd.DataFrame()
df = pd.DataFrame()
# ... intermediate steps with data manipulations ....
newdata = pd.concat([newdata,df],axis=0)  # axis=0 means concatenating by row


# https://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe
# Randomly sample n elements from your dataframe, & create a new combined df
d1_elements = d1.sample(n=5000)
d2_elements = d2.sample(n=5000)
dw_elements = dw.sample(n=10000)
dn = pd.DataFrame()
dn = pd.concat([dw_elements, d1_elements, d2_elements],ignore_index=True)


## ADD dataframes by column
newdata = pd.DataFrame()
df = pd.DataFrame()
# ... intermediate steps with data manipulations ....
newdata = pd.concat([newdata,df],axis=1)  # axis=1 means concatenating by column



####  merge  : inner join, outer join, etc
# https://pandas.pydata.org/pandas-docs/stable/merging.html
new_df = pd.merge(df1, df2, how='inner', on='ID')


## assigning data from one df (by row) to another
new_df.loc[0, 'dateTime'] = df.loc[(rownum), 'dateTime']



#### array (row) to df, 
df=pd.DataFrame(array).T     # T implies taking transpose


#### convert data from dataframe colunm to an array
data_array = pd.np.array(df.iloc[:,[colnum]]).reshape(len(df))



####   append, extend
corr = []
for i in range(len(predicted)):  ## looping over list
    corr.append(np.corrcoef(predicted[i], actual[i]['Y']))
    
fullact = []
for i in range(len(testsets)):     # testsets is a hash, where each hash element contains a list
    act = testsets[i]['Hb_level']
    #id = testsets[i]['ID']
    #pred = predicted[i]
    for j in range(len(act)):
        fullact.append(act[j])  # ([act[j],id[j],pred[j]])    

Q_thres = []
Q_thres.append([mc, w, q, np.median(temp.kWh_per_piece), np.median(temp.timePerPc)])
Q_thres = pandas.DataFrame(Q_thres, columns=['machineID','variantID', 'Q_threshold', 'median_kpi', 'timePerPc'])

        
        
####  loc (by column name) & iloc (by column no.)
prdcn.loc[:, ['Date','Shift','Machine','TotalProductPcs','kWh','TotalStop_minute']].head(n=5)
dataX = feature_cleaned.iloc[:,[0,1,2,4,5,6,7,8,9]].values     



####   passing df row to a function through loop
for i in range(0,df.shape[0]):  
    total=0
    X_score = user_defined_python_function(df.loc[i, 'columnX'])


    
## removing & adding row to df
backup = backup.drop(backup.head(1).index) #, inplace=True)  # remove row from top
backup = backup.append(df.iloc[[i]]) #, axis=0)  ## add row at end
    
  
  
### taking subsets : subset by row
df = df[df.colname == some_value]

## subset by column
df = df[['X1','X2','X4','Y']]  # by column name
# or,
df = df.loc[:, ['X1','X2','X4','Y']]

df = df.iloc[:,[0,1,3,4]].values     # by column no.


match = df[(df.y_test == df.ypred_svc) & (df.y_test == df.ypred_rf) & (df.y_test == df.ypred_gb) 
           & (df.y_test == df.ypred_mlp) & (df.y_test == df.ypred_knn) & (df.y_test == df.ypred_dt) ]

# obtain inverse subset of above; that is, rows not satisfying the conditions
# https://stackoverflow.com/questions/41800424/remove-rows-in-python-less-than-a-certain-value
df2 = df[~( (df.y_test == df.ypred_svc) & (df.y_test == df.ypred_rf) & (df.y_test == df.ypred_gb) 
           & (df.y_test == df.ypred_mlp) & (df.y_test == df.ypred_knn) & (df.y_test == df.ypred_dt) )]



"""  
mismatch = Pred[(Pred.y_test != Pred.y_pred_svc) & (Pred.y_test != Pred.y_pred_rf) & (Pred.y_test != Pred.y_pred_rf_opt) 
           & (Pred.y_test != Pred.y_pred_svc_opt) & (Pred.y_test != Pred.y_pred_knn) ]    

## get the remaining pediction cases
df2 = Pred[~( (Pred.y_test == Pred.y_pred_svc) & (Pred.y_test == Pred.y_pred_rf) & (Pred.y_test == Pred.y_pred_rf_opt) 
           & (Pred.y_test == Pred.y_pred_svc_opt) & (Pred.y_test == Pred.y_pred_knn) )]

df3 = df2[~( (df2.y_test != df2.y_pred_svc) & (df2.y_test != df2.y_pred_rf) & (df2.y_test != df2.y_pred_rf_opt) 
           & (df2.y_test != df2.y_pred_svc_opt) & (df2.y_test != df2.y_pred_knn) )]
"""
