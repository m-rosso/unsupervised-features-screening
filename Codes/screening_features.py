####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Class for screening continuous features based on variation criteria:

class VarScreeningNumerical(object):
    """
    Arguments for initialization:
        'features': list with names of numerical variables.
        'na_features': list with dummies for missing value status (optional).
        'stat': statistic for computing variability of a numerical variable. Choose among 'variance' and 'coefficient of variation',
        the ratio between standard deviation and mean.
        'select_k': indicates whether a given number of features should be selected.
        'k': number (integer or float) of numerical variables to be selected based on the variance criterion (p*).
        'thresholding': indicates whether only features with variance larger than a threshold should be selected.
        'variance_threshold': threshold for selecting features based on their variances.
        'winsorize': boolean indicating whether to use winsorized data or not.
        'winsorize_param': float (between 0 and 1) for the percentile to winsorize data.
        'drop_outliers': boolean indicating whether to drop outliers previously to variance calculation.
        'drop_outliers_param': float (between 0 and 1) for the percentile to drop outliers.
        'collinearity': boolean indicating whether to apply the multicollinearity filter.
        'collinearity_param': float (between 0 and 1) for the threshold against which the R2 should be compared to.
    Methods:
        'select_feat': for a given dataframe ('data'), performs selection of numerical variables based on variance criterium and using
        parameters passed during initialization.
        'coef_variation': static method that returns the coefficient of variation for a numerical variable.
    Output objects:
        'features': list of numerical variables for selection.
        'selected_feat': list with names of selected features.
        'no_var_continuous_feat': list with names of features with zero variance.
        'var_continuous_feat': dataframe with variance on training data by numerical variable.
    """
    def __init__(self, features,
                 select_k=True, k=None, thresholding=False, variance_threshold=0,
                 na_features=[], stat='variance',
                 winsorize=False, winsorize_param=0.025, drop_outliers=False, drop_outliers_param=0.01,
                 collinearity=False, collinearity_param=0.9):
        self.features = features
        self.num_all_continuous = len(features)
        self.na_features = na_features
        self.stat = stat
        self.select_k = select_k
        self.k = k
        self.thresholding = thresholding
        self.variance_threshold = variance_threshold
        self.winsorize = winsorize
        self.winsorize_param = winsorize_param
        self.drop_outliers = drop_outliers
        self.drop_outliers_param = drop_outliers_param
        self.collinearity = collinearity
        self.collinearity_param = collinearity_param
        
    # Method that implements the screening of features based on variance:
    def select_feat(self, data):
        # Variance by feature:
        if self.stat == 'coef_variation':
            var_continuous_feat = pd.DataFrame(data={'feature': list(self.features),
                                                     'variation': [self.coef_variation(data[f]) for f in
                                                                   self.features]})
        
        else:
            var_continuous_feat = pd.DataFrame(data={'feature': list(self.features),
                                                     'variation': [np.nanvar(data[f]) for f in self.features]})
        
        # Features with no variance:
        self.no_var_continuous_feat = list(var_continuous_feat[var_continuous_feat.variation == 0].feature)
        
        # Features with positive variance:
        var_continuous_feat = var_continuous_feat[~(var_continuous_feat.feature.isin(self.no_var_continuous_feat))]

        # Dummy indicating if there is feature for missing value:
        var_continuous_feat['missing'] = var_continuous_feat.feature.apply(lambda x: 1 if 'NA#' + x in self.na_features else 0)
        
        # Sorting features by variance:
        self.var_continuous_feat = var_continuous_feat.sort_values('variation', ascending=False)
        self.var_continuous_feat.columns.name = self.stat
        
        data = data.copy()
        
        # Winsorizing data:
        if self.winsorize:    
            for f in self.features:
                delta_neg = np.quantile(data[f], q=self.winsorize_param)
                delta_pos = np.quantile(data[f], q=(1-self.winsorize_param))

                data[f] = data[f].apply(lambda x: min([delta_pos, max([delta_neg, x])]))
        
        # Dropping outliers:
        if self.drop_outliers:
            for f in self.features:
                perc_inf = np.quantile(data[f], q=self.drop_outliers_param)
                perc_sup = np.quantile(data[f], q=(1-self.drop_outliers_param))
                
                data[f] = data[f].apply(lambda x: x if (x < perc_sup) & (x > perc_inf) else np.NaN)
        
        if self.select_k:
            self.__select_k_feats(data=data)
        
        if self.thresholding:
            self.__thresholding_selection(data=data)
        
    # Method that selects those features with the K highest variability:
    def __select_k_feats(self, data):
        # Selecting features with the highest variance:
        self.selected_feat = []

        # Selection of features considering the association between them:
        if self.collinearity:
            self.__collinearity_handle(data=data)
            
        else:        
            # Loop over features:
            for i in range(len(self.var_continuous_feat)):
                # Check if enough features have already been selected:
                if len(self.selected_feat) < self.k:
                    # Check if there is dummy variable for missing value:
                    if self.var_continuous_feat.missing.iloc[i] == 1:
                        self.selected_feat.append(self.var_continuous_feat.feature.iloc[i])
                        self.selected_feat.append('NA#' + self.var_continuous_feat.feature.iloc[i])
                    else:
                        self.selected_feat.append(self.var_continuous_feat.feature.iloc[i])

    def __thresholding_selection(self, data):
        # Selecting features with the highest variance:
        self.selected_feat = [f for f in self.features if data[f].var() > self.variance_threshold]
        
        # Check if there is dummy variable for missing value:
        self.selected_feat.extend(['NA#' + f for f in self.selected_feat if 'NA#' + f in self.na_features])
                        
    def __collinearity_handle(self, data):
        # Initializing by the feature with the highest variance:
        sel_feats = [list(self.var_continuous_feat.feature)[0]]
        self.selected_feat.append(list(self.var_continuous_feat.feature)[0])
        if 'NA#' + list(self.var_continuous_feat.feature)[0] in self.na_features:
            self.selected_feat.append('NA#' + list(self.var_continuous_feat.feature)[0])

        # Loop over features:
        for f in list(self.var_continuous_feat.feature)[1:]:
            # Check if enough features have already been selected:
            if len(self.selected_feat) < self.k:
                pass
            else:
                break

            candidate_feat = f
            print('-------------------------------------------------------------------------')
            print('Number of selected features: {}'.format(len(sel_feats)))
            print('Candidate feature: {}'.format(candidate_feat))

            # Create estimation object:
            model = LinearRegression()

            # Fitting linear regression model:
            model.fit(data[sel_feats], data[candidate_feat])

            # Calculating R2 between candidate feature and those already selected:
            y_hat = list(model.predict(data[sel_feats]))
            avg_y = np.mean(data[candidate_feat])
            sqr = sum([(i - j)**2 for i, j in zip(data[candidate_feat], y_hat)])
            sqt = sum([(i - avg_y)**2 for i in data[candidate_feat]])
            R2 = 1 - sqr/sqt
            print('R2 from regression of candidate feature against selected features: {}'.format(round(R2, 4)))

            # Condition for selecting a feature based on its association with already selected features:
            if R2 <= self.collinearity_param:
                sel_feats.append(candidate_feat)
                self.selected_feat.append(candidate_feat)

                # Check if there is dummy variable for missing value:
                if 'NA#' + candidate_feat in self.na_features:
                    self.selected_feat.append('NA#' + candidate_feat)
                print('\033[1mCandidate feature selected!\033[0m')

            else:
                print('\033[1mCandidate feature not selected!\033[0m')

            print('-------------------------------------------------------------------------')
            print('\n')

    # Function that returns the coefficient of variation for a numerical variable:
    @staticmethod
    def coef_variation(data):
        """
        Arguments:
            'data': series, array, or list with numerical data.
        Outputs:
            'coef_var': the coefficient of variation for passed data.
        """
        try:
            coef_var = np.nanstd(data)/np.nanmean(data)
        except:
            coef_var = np.NaN

        return coef_var

####################################################################################################################################
# Class for screening categorical features:

class VarScreeningCategorical(object):
    """
    Arguments for initialization:
        'features': list of categorical features whose categories should be selected.
        'variance_param': parameter for selection based on the variance of a given dummy variable.
    Methods:
        'select_feat': for a given dataframe ('data'), performs selection of dummies based on variance criterium.
    Output objects:
        'self.features': list of categorical features whose categories should be selected.
        'self.variance_param': parameter for selection based on the variance of a given dummy variable.
        'self.dummies': dataframe with selected dummies for passed data.
        'self.categorical_info': dictionary with number of overall categories, number of selected categories, and selected
        categories for each categorical feature.
    """
    def __init__(self, features, variance_param = 0.01):
        self.features = features
        self.variance_param = variance_param
        
    # Method that implements the screening of features based on variance:
    def select_feat(self, data):
        # Dictionary with information on categorical features:
        self.categorical_info = {}

        # Dataframe with dummy variables:
        self.dummies = pd.DataFrame(data=[])

        # Loop over categorical features:
        for f in cat_vars:
            self.categorical_info[f] = {}

            # Creating dummy variables:
            dummies_cat = pd.get_dummies(data[f])
            self.categorical_info[f]['all_categories'] = dummies_cat.shape[1]

            # Selecting categories depending on their variance:
            selected_cat = [d for d in dummies_cat.columns if dummies_cat[d].var() > self.variance_param]
            self.categorical_info[f]['selected_categories'] = selected_cat

            # Dataframe with dummy variables for a given categorical feature:
            dummies_cat = dummies_cat[selected_cat]
            dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

            # Dataframe with dummy variables for all categorical features:
            self.dummies = pd.concat([self.dummies, dummies_cat], axis=1)
