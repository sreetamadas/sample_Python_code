# https://stackoverflow.com/questions/30522724/take-multiple-lists-into-dataframe
# https://stackoverflow.com/questions/30631841/pandas-how-do-i-assign-values-based-on-multiple-conditions-for-existing-columns

gender = ['male','male','male','female','female','female','squirrel']

pet1 =['dog','cat','dog','cat','dog','squirrel','dog']

pet2 =['dog','cat','cat','squirrel','dog','cat','cat']

d = pd.DataFrame(np.column_stack([gender, pet1, pet2]),columns=['gender', 'petl', 'pet2'])

d

d['points'] = np.where( ( (d['gender'] == 'male') & (d['pet1'] == d['pet2'] ) ) | ( (d['gender'] == 'female') & (d['pet1'].isin(['cat','dog'] ) ) ), 5, 0)

