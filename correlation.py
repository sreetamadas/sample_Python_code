from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau

x = [1, 2, 4, 7]
y = [1, 3, 4, 8]

pearsonr(x, y)
# output: SpearmanrResult(correlation=0.9843, pvalue=0.01569)

spearmanr(x, y)
# output: SpearmanrResult(correlation=1.0, pvalue=0.0)

kendalltau(x, y)
# output: KendalltauResult(correlation=1.0, pvalue=0.0415)



## compute correlation matrix with selected columns (x1, x2, x3) of a dataframe, & by different methods
corr_p = df[['X1', 'x2', 'x3']].corr(method='pearson')  # pearson, spearman
print(corr_p)
