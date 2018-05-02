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
