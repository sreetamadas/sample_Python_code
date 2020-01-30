#####  dataframe: creation &  manipulations  #####

##########   CREATION   ##############
gender = ['male','male','male','female','female','female','squirrel']
pet1 =['dog','cat','dog','cat','dog','squirrel','dog']
pet2 =['dog','cat','cat','squirrel','dog','cat','cat']

d = pd.DataFrame(np.column_stack([gender, pet1, pet2]),columns=['gender', 'petl', 'pet2'])
d['points'] = np.where( ( (d['gender'] == 'male') & (d['pet1'] == d['pet2'] ) ) | 
                       ( (d['gender'] == 'female') & (d['pet1'].isin(['cat','dog'] ) ) ), 5, 0)


##################   MANIPULATIONS   #################
####  concatenation  ####
# ADD 1 df below the other
newdata = pd.DataFrame()
df = pd.DataFrame()
# ... intermediate steps with data manipulations ....
newdata = pd.concat([newdata,df],axis=0)  # axis=0 means concatenating by row

## ADD dataframes by column
newdata = pd.DataFrame()
df = pd.DataFrame()
# ... intermediate steps with data manipulations ....
newdata = pd.concat([newdata,df],axis=1)  # axis=1 means concatenating by column



####  merge  : inner join, outer join, etc
# https://pandas.pydata.org/pandas-docs/stable/merging.html
new_df = pd.merge(df1, df2, how='inner', on='ID')



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


    
### taking subsets    
match = df[(df.y_test == df.ypred_svc) & (df.y_test == df.ypred_rf) & (df.y_test == df.ypred_gb) 
           & (df.y_test == df.ypred_mlp) & (df.y_test == df.ypred_knn) & (df.y_test == df.ypred_dt) ]

# obtain inverse subset of above; that is, rows not satisfying the conditions
# https://stackoverflow.com/questions/41800424/remove-rows-in-python-less-than-a-certain-value
df2 = df[~( (df.y_test == df.ypred_svc) & (df.y_test == df.ypred_rf) & (df.y_test == df.ypred_gb) 
           & (df.y_test == df.ypred_mlp) & (df.y_test == df.ypred_knn) & (df.y_test == df.ypred_dt) )]




    
    
