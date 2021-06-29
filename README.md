## Unsupervised screening of features

This repository presents codes and results of experiments constructed upon the idea of pre-selecting features for data modeling based on their variances. This constitutes a *unsupervised features screening*, since the response variable is not used during the selection. Not looking to the response variable turns possible the use of the entire training data without any validation procedure (under stationarity assumptions), which would complexify the analysis even further, since cross-validation is usually applied for choosing hyper-parameters of the estimation method, while train-test split is done for obtaining a relieful estimate of predictive accuracy.
<br>
<br>
Given the assumption that input variables with higher variance better depict the expected diversity to be found on unseen data points, features with the highest variances can be seen as good candidates to be the most important features during model estimation and generalization. This is specially relevant for learning problems with *high-dimensional datasets*, or for which it is expected a *high noise-to-signal ratio*.
<br>

### Expected benefits
Among the benefits of a variance-based features screening, the most likely to hold is the *reduction in running time*, since less parameters would be estimated (for parametric models) or the learning method would search across a smaller feature space (for non-parametric models). By the exclusion of less relevant features, the noise-to-signal ratio of the model can be reduced, thus improving its ability to generalize. This is specially due to the smaller variance of estimates, which *reduces the variance component of the test error*. Consequently, both the average and the variance of performance metrics can be improved by a relatively costless procedure of features selection.
<br>

### Data types
It is crucial to stress that two distinct screening procedures is available: one for numerical and other for categorical data, where the last one covers only variance thresholding.
<br>
Concerning data types for implementing the code, it was constructed upon Numpy and Pandas libraries, instead of just Numpy. This makes implementation dependent on converting data structure into dataframes, instead of more general possibilities. No complex modifications would be necessary to generalize the data structure, and nothing that would change results.
<br>

### Proposed variance-based screening of features
It starts with the descendent sorting of features according to their variances calculated on the training data. Then, from all *p* original features, only those *p* < p* with the highest variances are selected. This default implementation can be improved by data pre-processing operations that refine the screening of features. The following alternatives were developed:
* **Winsorized data:** replaces values below and above the percentile of d% and (1 - d)% by these values, respectively. This works to attenuate the influence of extreme values that may distort the variance calculation.
* **Drop of outliers:** directly removes outliers from the sample previous to the variance calculation.
* **Multicollinearity filter:** after sorting numerical variables in a descendent order according to their variances, an input *X*  is only selected if its correlation with previously selected inputs is below a given threshold. This correlation is calculated by regressing the candidate input against all previously selected inputs, and then by measuring the R2 coefficient. If R2 < thres, then X can be selected. The procedure continues as long as the number of selected inputs is smaller than *p**.
<br>

* **Variance thresholding**: another possibility covered by the developed class is the variance thresholding, where instead of defining the number of features to be selected, only variables with variance higher than a threshold will be selected.

### Experiments for tests
In order to assess the impacts on performance metrics and running time of unsupervised screening of features based on variance, 70 different high-dimensional datasets (N < p) for binary classification task provided the empirical background for experiments. Then, with and without the implementation of variance-based features screening, 1000 bootstrap estimations were conducted for data modeling using two distinct learning methods (logistic regression and GBM). Finally, the computation of statistics for performance metrics guided conclusions towards different screening options.
<br>

### Results of tests
* The variance-based unsupervised screening of features has improved the predictive accuracy of *logistic regression models*. An estimation time reduced by a factor of 4 is another advantage of the procedure proposed here to deal with high-dimensionality.
* Even that the implemenation of unsupervised screening of features has not make *GBMs* generalize better on average, these models have significantly more stable performances with a reduced model complexity. The reduction on expected running time is a key contribution to GBMs, since they usually take a long time to be estimated.

### Contents of this repository
* Folder *Codes*: contains Python scripts for running tests and to implement the proposed variance-based screening of features. Besides, it also presents a HTML file with demonstration. *Mainly, the module screening_features presents the classes VarScreeningNumerical and VarScreeningCategorical for applying features selection based on variance, for both numerical and categorical data, respectively*.
* Folder *Results*: presents HTML files with more discussion on the screening of features and with the results of tests.
