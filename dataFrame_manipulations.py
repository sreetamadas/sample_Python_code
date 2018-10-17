#####  dataframe manipulations  #####


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



####   append, extend
corr = []
for i in range(len(predicted)):  ## looping over list
    corr.append(np.corrcoef(predicted[i], actual[i]['Y']))
    
fullact = []
for i in range(len(testsets)):
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


    
