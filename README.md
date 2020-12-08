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

<br>
<br>
### Experiments for tests
In order to assess the impacts on performance metrics and running time of unsupervised screening of features based on variance, 70 different datasets for binary classification task provided the empirical background for experiments that take on the following steps:
1. Selection of high-dimensional datasets (N < p).
<br>
2. Screening of features based on variance (when this procedure is implemented).
<br>
3. Data pre-processing (numerical data transformation, categorical data transformation, assessment and treatment of missing values).
<br>
4. 1000 bootstrap estimations for data modeling using two distinct learning methods: logistic regression and GBM.
<br>
    * Estimations were divided into two: K-folds CV using training data to define hyper-parameters, and train-test estimation to assess performance metrics.
<br>
5. Computation of statistics for performance metrics.
    * Average and standard deviation over 1000 bootstrap estimations of ROC-AUC, average precision score, and Brier score.
<br>
6. Comparison between statistics with and without features screening, and for different screening options.
<br>
<br>
### Results of tests

Prévia da estrutura (ordenamento decrescente das variáveis de acordo com a sua variância, opções de especificação - remoção de outliers, winsorized data, filtro para variáveis colineares), dos resultados (média e desvio padrão de performance metrics evaluated on test data, running time). Estrutura do código (data types, etc.).
