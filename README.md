## Unsupervised screening of features

This repository presents codes and results of experiments constructed upon the idea of pre-selecting features for data modeling based on their variances. This constitutes a *unsupervised features screening*, since the response variable is not used during the selection. Not looking to the response variable turns possible the use of the entire training data without any validation procedure (under stationarity assumptions), which would complexify the analysis even further, since cross-validation is usually applied for choosing hyper-parameters of the estimation method, while train-test split is done for obtaining a relieful estimate of predictive accuracy.
<br>
<br>
Given the assumption that input variables with higher variance better depict the expected diversity to be found on unseen data points, features with the highest variances can be seen as good candidates to be the most important features during model estimation. This is specially relevant for learning problems with *high-dimensional datasets*, or for which it is expected a *high noise-to-signal ratio*.
<br>
<br>
### Expected benefits
Among the benefits of a variance-based features screening, the most likely to hold is the *reduction in running time*, since less parameters would be estimated (for parametric models) or the learning method would search across a smaller feature space (for non-parametric models). By the exclusion of less relevant features, the noise-to-signal ratio of the model can be reduced, thus improving its ability to generalize. This is specially due to the smaller variance of estimates, which *reduces the variance component of the test error*. Consequently, both the average and the variance of performance metrics can be improved by a relatively costless procedure of features selection.
<br>
<br>
### Proposed variance-based screening of features

Prévia da estrutura (ordenamento decrescente das variáveis de acordo com a sua variância, opções de especificação - remoção de outliers, winsorized data, filtro para variáveis colineares), dos experimentos (70 datasets de classificação binária; 1000 estimações de bootstrap com e sem variance-based screening of features; logistic regression and GBM) dos resultados (média e desvio padrão de performance metrics evaluated on test data, running time). Estrutura do código (data types, etc.).
$\beta$
