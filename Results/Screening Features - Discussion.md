
## Screening features for high-dimensional datasets
## Discussion

**Introduction**

Any dataset is conventionally defined along different observations, accounting for $N$, and a set of features characterizing each one of them, totalizing $p$ different input variables $X$, plus a response variable $Y$, for  supervised learning problems. Therefore, a dataset constitutes a $N x K$ matrix, where $K = p + 1$ (for the intercept). The objective is to use this available data to approximate a target function $F(X)$ that deterministically relates output and input variables. Supposing an additive random error term, $\epsilon$, $Y$ is defined by:
\begin{equation}
Y = F(X) + \epsilon
\end{equation}
For major empirical contexts, $N > p$, which is precisely what is required by some standard statistical learning methods. Even when not required, having more observations than inputs may help achieving more stable results from model estimation. In any case, advances in data collection and storage increase the possibility of working with **high-dimensional datasets**, for which $p \gg N$.
<br>
<br>
If a large collection of features improves the characterization of observations, and thus the expected capacity of explaning the response variable, in the other hand it may introduce inputs whose contribution is questionable, specially when the number of such inputs is very high (*curse of dimensionality*). This can be measured and understood in terms of the *signal-to-noise ratio (SNR)*, the proportion of variability in $Y$ that follows from variations in $X$ in terms of variations in $\epsilon$. The larger the dimensionality of $X$, the smaller one should expect the SNR to be.
<br>
<br>
Despite of the existence of approaches proper for this context of more features than observations, such as principal components analysis (PCA) and other approaches presented in chapter 18 of Friedman, Hastie, and Tibshirani (2008),  one simple possibility is to perform a previous *unsupervised screening of features*. Not looking to the response variable turns possible the use of the entire training data without any validation procedure, which would complexify the analysis even further, since cross-validation is usually applied for choosing hyper-parameters of the estimation method, while train-test split is done for obtaining a relieful estimate of predictive accuracy.
<br>
<br>
The advantages of such unsupervised features screening go beyond making possible the implementation of basic estimation methods. There is also the possibility of reducing variance of estimation outputs, which will be assessed and discussed through a methodology presented next. Furthermore, decreasing data dimensionality certainly reduces running time, benefit specially useful when large scale tests are demanded by any study that uses high-dimensional datasets.

------------

**Suggested procedure for unsupervised selection of features**

Given the assumption that input variables with higher variance better depict the expected diversity to be found on unseen data points, from a given set of variables $X = \{X_1, X_2, ..., X_p\}$, those with the $p^*$-highest variances, where $p^* < p$, can be seen as good candidates to be the most important features during model estimation, at least from a unsupervised perspective.
<br>
The implementation of this unsupervised features screening depends on the definition of $p^*$. The method proposed and tested here considers the following setting for choosing $p^*$. Suppose a collection of datasets $S = \{s_1, s_2, ..., s_S\}$, where $s \in S$ is a matrix $N_s x p_s$ and whose data generation processes are somewhat similar - for instance, each dataset may have a same response variable, though referencing a different population.
<br>
The ratio between number of observations and number of input variables of a dataset $s$ is given by $k_s = N_s/p_s$. So, from the collection $S$, the following two subsets can be established:
\begin{equation}
S' = \{s | k_s \leq 1\}
\end{equation}
<br>
\begin{equation}
S^* = \{s | k_s > 1\}
\end{equation}
<br>
Where $S'$ defines the subset of high-dimensional datasets and $S^*$ the subset of datasets for which there are more observations than features. The goal is to use the average or the minimum $k_s$ for $s \in S^*$ in order to define a $\overline{k}$ of reference for stores on $S'$. Once $\overline{k}$ is defined from subset $S^*$, then $p_s^*$, for $s \in S'$, can be defined by:
\begin{equation}
p_s^* = floor\Big(\frac{N_s}{\overline{k}}\Big)
\end{equation}
<br>
The unsupervised screening of features continues by ordering the set $X_s$ using the variance of each variable on training data, so that emerges a subset $X_s^* = \{X_{v_1}^s, X_{v_2}^s, ..., X_{v_{p_s^*}}^s\}$ of the $p_s^*$ variables with the highest variances.
<br>
This procedure demands the existence of a collection of datasets $S$. A more general approach, therefore, is to consider the following definition of $p_s^*$:
<br>
<br>
\begin{equation}
\displaystyle p_s^* = \alpha*N_s
\end{equation}
Where $\alpha \in (0, 1]$. The hyper-parameter $\alpha$ may be chosen through cross-validation, or based on ad hoc reasoning. By doing this, a dataset with filtered features would help model estimation as features more likely to be useful for prediction are provided.
<br>
<br>
Above, the subset $X_s^*$ of input variables with the highest variances were defined using only a variability criterion. Improved procedures can make the screening of features more robust to extreme values and help fitted models to generalize better. The following alternatives were developed and used during experiments:
1. **Winsorized data:** when a numerical variable $X_j$ gets winsorized, its values are redefined as follows: $x_j^{new} = \min\{\delta_+, \max\{\delta_-, x_j^{old}\}\}$, where $\delta_+$ is the percentile $1 - \beta$ and $\delta_-$ is the percentile $\beta$, where $\beta \in (0,1)$. Therefore, extremely high values of $X_j$ are turned into the percentile $1 - \beta$, while extremely low values are turned into the percentile $\beta$. Consequently, winsorizing data works removing the influence of outliers.
2. **Drop of outliers:** before calculating the variance of inputs, outlier observations can be removed from the sample. A simple setting would exclude values above or below a given extreme percentile, such as 2.5%. Thus, the drop of outliers explicitly removes the influence of such extreme observations.
3. **Multicollinearity filter:** after sorting numerical variables in a descendent order according to their variances, an input $X_j$ is only selected if its correlation with previously selected inputs is below a threshold. This correlation is calculated by regressing the candidate input against all previously selected inputs, and then by measuring the $R^2$ coefficient. If $R^2_j < thres$, then $X_j$ can be selected. The procedure continues as long as the number of selected inputs is smaller than $p_s^*$.

In order to test whether using $X_s^*$ instead of $X_s$ as the set of original inputs for estimating a model for $s \in S'$ reduces the variance of its estimation, $T$ bootstrap estimations using $X_s$ and $T$ bootstrap estimations using $X_s^*$ will be performed for a sample of different high-dimensional datasets. Next, statistics of performance metrics will guide the conclusions with respect to this proposed unsupervised screening of features. Besides, running times for each dataset will also be compared to denote how fast estimations are using the selected set of inputs.

----------------

**Data types**

It is crucial to stress that the above discussion applies for numerical, or continuous input variables. Categorical and binary data should be treated aside, since their variances assume values quite different from numerical variables. Consequently, two distinct screening procedures can be performed: one for numerical and other for categorical data.
<br>
The screening of categorical data implemented here does not select categorical inputs per se, but rather selects only those dummy variables whose variance on training data surpasses a given threshold, such as $0.01$. This strategy excludes categories that either occurr too few or too much. **So, being $p_s^*$ the number of inputs to be selected and $p_s^d$ the number of dummies kept by the screening of categorical data, then only $p_s^* - p_s^d$ numerical inputs were selected according to their variance**.
<br>
The experiments designed for this study preserves all categorical and binary variables, and only selects dummies with sufficient variability. This may represent an issue for empirical contexts with a high number of non-numerical inputs. For some of the datasets used here, however, this was precisely the case, since some of them has so few observations that the defined number of features to be selected, $p_s^*$, was smaller than the total number of categorical and binary inputs, so $p_s^* - p_s^d < 0$. When this happened, the approach was to keep all dummies and select $p_s^*$ numerical features, instead of $p_s^* - p_s^d$. 

A final remark concerns libraries needed for implementing the suggested unsupervised screening of features. For the sake of easiness of development, codes are constructed upon Numpy and Pandas libraries, instead of just Numpy. This makes implementation dependent on converting data structure into dataframes, instead of more general possibilities. No complex modifications would be necessary to generalize the data structure, and nothing that would change results.

------------

**Experiments design and methodology**

In order to assess the impacts on performance metrics and running time of unsupervised screening of features based on variance, about 70 different datasets for binary classification task provided the empirical background for experiments that take on the following steps:
1. Selection of high-dimensional datasets (N < p).
2. Screening of features based on variance (when this procedure is implemented).
3. Data pre-processing (numerical data transformation, categorical data transformation, assessment and treatment of missing values).
4. Bootstrap estimations for data modeling using two distinct learning methods: logistic regression and GBM.
    * Estimations were divided into two: K-folds CV using training data to define hyper-parameters, and train-test estimation to assess performance metrics.
5. Computation of statistics for performance metrics.
    * Average and standard deviation over 1000 bootstrap estimations of ROC-AUC, average precision score, and Brier score.
6. Comparison between statistics with and without features screening, and for different screening options.

The tests consist on two collections of estimations, one for logistic regression model and the other for GBM, while each one of them has 5 rounds of estimations for all 70 datasets:
* Round 1: no unsupervised screening of features based on variance. This case involves high-dimensional datasets.
* Round 2: unsupervised screening of features based on variance and using the default implementation of screening.
* Round 3: uses winsorized data to produce results robust to outliers.
* Round 4: drops outliers when calculating variances, also to produce results robust to outliers.
* Round 5: adds to the variance criterion of selection a method for filtering out features excessively correlated with previously selected features.

Concerning methodological details, although data pre-processing procedures are straightforward here, some aspects of missing values treatment should be detached. After log-transforming and standard scaling numerical data, missing values (from numerical or binary inputs) are filled in by zeros. Moreover, each input $X_j$ with missing values generate a binary variable for missing value status $NA\#X_j$. With respect to the implementation of unsupervised screening of features based on variance, nothing significantly changes. The only modification is that additional binary inputs are created, amplifying the high-dimensionality issue.

----------------
<a id='main_results'></a>

**Main results**

1. **Logistic regression:** the variance-based unsupervised screening of features has improved the predictive accuracy of logistic regression models. Both average results and distribution of results have shown that models estimated using a subset of features with the largest variances are able to generalize better. An estimation time reduced by a factor of 4 is another advantage of the procedure proposed here to deal with high-dimensionality.
    * **Aggregated findings:**
        * Means (overall datasets) quite larger for the average test ROC-AUC and quite smaller for the standard deviation of test ROC-AUC when unsupervised screening of features is implemented. The same applies for the standard deviation of average precision score.
        * A measure that considers together average and standard deviation of test ROC-AUC and of test average precision score, given by the ratio, for each dataset and for each performance metric, between the average and the standard deviation, shows that the use of unsupervised screening of features based on their variance clearly improves predictive accuracy.
        * Unsupervised screening of features developed through outliers dropping has the better performance among tested alternatives.
        * Overall datasets, the average total running time reduces by a factor of 4 with the variance-based unsupervised screening of features.
    * **Distribution of outcomes:**
        * Unsupervised screening of features has improved distributions of average and standard deviation of test ROC-AUC. The same applies for average total running time, with less impressive results concerning average precision score.
        * Similar distributions of outcomes for different options of unsupervised screening of features. However, the drop of outliers slightly outperforms the others.
<br>
<br>
2. **GBM:** even that the implemenation of unsupervised screening of features has not make GBMs generalize better on average, these models have performed significantly more stable with the help of pre-selected sets of features that reduce model complexity by cutting off noise variables. The reduction on expected running time is a key contribution to GBMs, since they usually take a long time to be estimated.
    * **Aggregated findings:**
        * Standard deviations of test ROC-AUC and test average precision score are smaller with the implementation of features screening.
        * The average ratio between average and standard deviation of test ROC-AUC is considerably larger with variance-based screening of features.
        * Average total running time without features screening is 4 times larger than that necessary with features screening.
    * **Distribution of outcomes:**
        * Similar distributions, with and without the implementation, for the average and the standard deviation of test ROC-AUC and test average precision score.
        * The unsupervised screening of features based on variance and implemented using drop of outliers has a particularly good performance when compared to the benchmark (no screening).


```python

```
