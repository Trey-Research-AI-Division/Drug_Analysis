#!/usr/bin/env python3

from typing import Dict
from typing import NamedTuple
import kfp.dsl as dsl

import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath
import kfp.compiler as compiler

def download_data(url: str, output_text_path: OutputPath(str)):
    import requests

    req = requests.get(url)
    url_content = req.content

    with open(output_text_path, 'wb') as writer:
        writer.write(url_content)

# Gratefully adapted from Serigne Cisse's awesome notebook here
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
#
# If you read this and it benefits you, please go upvote that notebook.
def train(train_data: InputPath(),
          test_data: InputPath(),
          mlpipeline_metrics_path: OutputPath('Metrics'),
          output_path: OutputPath(str)):
    from scipy.special import boxcox1p
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import warnings
    from subprocess import check_output
    from scipy import stats
    from scipy.stats import norm, skew  # for some statistics
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
    from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    import json
    import lightgbm as lgb
    import xgboost as xgb

    def ignore_warn(*args, **kwargs):
        pass

    # ignore annoying warning (from sklearn and seaborn)
    warnings.warn = ignore_warn

    # Limiting floats output to 3 decimal points
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

    # Now let's import and put the train and test datasets in pandas dataframe
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    # Check the numbers of samples and features
    print("The train data size before dropping Id feature is : {} ".format(train.shape))
    print("The test data size before dropping Id feature is : {} ".format(test.shape))

    # Save the 'Id' column
    train_ID = train['Id']
    test_ID = test['Id']

    # Now drop the  'Id' colum since it's unnecessary for  the prediction process.
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)

    # Check again the data size after dropping the 'Id' variable
    print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
    print("The test data size after dropping Id feature is : {} ".format(test.shape))

    # Data Processing

    # Deleting outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) &
                       (train['SalePrice'] < 300000)].index)

    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    train["SalePrice"] = np.log1p(train["SalePrice"])

    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    print("all_data size is : {}".format(all_data.shape))

    # Missing Data

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(
        all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    missing_data.head(20)

    # ###Imputing missing values

    # We impute them  by proceeding sequentially  through features with missing values

    # - **PoolQC** : data description says NA means "No  Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

    # - **MiscFeature** : data description says NA means "no misc feature"
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

    # - **Alley** : data description says NA means "no alley access"
    all_data["Alley"] = all_data["Alley"].fillna("None")

    # - **Fence** : data description says NA means "no fence"
    all_data["Fence"] = all_data["Fence"].fillna("None")

    # - **FireplaceQu** : data description says NA means "no fireplace"
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    # - **LotFrontage** : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can **fill in missing values by the median LotFrontage of the neighborhood**.
    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    # - **GarageType, GarageFinish, GarageQual and GarageCond** : Replacing missing data with None
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')

    # - **GarageYrBlt, GarageArea and GarageCars** : Replacing missing data with 0 (Since No garage = no cars in such garage.)
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)

    # - **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath** : missing values are likely zero for having no basement
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    # - **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2** : For all these categorical basement-related features, NaN means that there is no  basement.
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')

    # - **MasVnrArea and MasVnrType** : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    # - **MSZoning (The general zoning classification)** :  'RL' is by far  the most common value.  So we can fill in missing values with 'RL'
    all_data['MSZoning'] = all_data['MSZoning'].fillna(
        all_data['MSZoning'].mode()[0])

    # - **Utilities** : For this categorical feature all records are "AllPub", except for one "NoSeWa"  and 2 NA . Since the house with 'NoSewa' is in the training set, **this feature won't help in predictive modelling**. We can then safely  remove it.
    all_data = all_data.drop(['Utilities'], axis=1)

    # - **Functional** : data description says NA means typical
    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    # - **Electrical** : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    all_data['Electrical'] = all_data['Electrical'].fillna(
        all_data['Electrical'].mode()[0])

    # - **KitchenQual**: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent)  for the missing value in KitchenQual.
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(
        all_data['KitchenQual'].mode()[0])

    # - **Exterior1st and Exterior2nd** : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(
        all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(
        all_data['Exterior2nd'].mode()[0])

    # - **SaleType** : Fill in again with most frequent which is "WD"
    all_data['SaleType'] = all_data['SaleType'].fillna(
        all_data['SaleType'].mode()[0])

    # - **MSSubClass** : Na most likely means No building class. We can replace missing values with None
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    # Is there any remaining missing value ?
    # Check remaining missing values if any
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(
        all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    missing_data.head()

    # ###More features engeneering

    # **Transforming some numerical variables that are really categorical**

    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    # Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    # Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # **Label Encoding some categorical variables that may contain information in their ordering set**
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # shape
    print('Shape all_data: {}'.format(all_data.shape))

    # **Adding one more important feature**
    # Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house

    # Adding total sqfootage feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + \
        all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # **Skewed features**
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(
        lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness.head(10)

    # **Box Cox Transformation of (highly) skewed features**

    # We use the scipy  function boxcox1p which computes the Box-Cox transformation of **\\(1 + x\\)**.
    #
    # Note that setting \\( \lambda = 0 \\) is equivalent to log1p used above for the target variable.
    #
    # See [this page][1] for more details on Box Cox Transformation as well as [the scipy function's page][2]
    # [1]: http://onlinestatbook.com/2/transformations/box-cox.html
    # [2]: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.special.boxcox1p.html

    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(
        skewness.shape[0]))

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    all_data = pd.get_dummies(all_data)
    print(all_data.shape)

    train = all_data[:ntrain]
    test = all_data[ntrain:]

    # #Modelling

    # **Define a cross validation strategy**

    # We use the **cross_val_score** function of Sklearn. However this function has not a shuffle attribut, we add then one line of code,  in order to shuffle the dataset  prior to cross-validation
    # Validation function
    n_folds = 5

    def rmsle_cv(model):
        kf = KFold(n_folds, shuffle=True,
                   random_state=42).get_n_splits(train.values)
        rmse = np.sqrt(-cross_val_score(model, train.values, y_train,
                       scoring="neg_mean_squared_error", cv=kf))
        return(rmse)

    # ##Base models

    # -  **LASSO  Regression**  :
    #
    # This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's  **Robustscaler()**  method on pipeline
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

    # - **Elastic Net Regression** :
    #
    # again made robust to outliers
    ENet = make_pipeline(RobustScaler(), ElasticNet(
        alpha=0.0005, l1_ratio=.9, random_state=3))

    # - **Kernel Ridge Regression** :
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

    # - **Gradient Boosting Regression** :
    #
    # With **huber**  loss that makes it robust to outliers
    #
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)

    # - **XGBoost** :
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=7, nthread=-1)

    # - **LightGBM** :
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    # ###Base models scores

    # Let's see how these base models perform on the data by evaluating the  cross-validation rmsle error
    score = rmsle_cv(lasso)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmsle_cv(ENet)
    print("ElasticNet score: {:.4f} ({:.4f})\n".format(
        score.mean(), score.std()))

    score = rmsle_cv(KRR)
    print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(
        score.mean(), score.std()))

    score = rmsle_cv(GBoost)
    print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(
        score.mean(), score.std()))

    score = rmsle_cv(model_xgb)
    print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmsle_cv(model_lgb)
    print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

    # ##Stacking  models

    # ###Simplest Stacking approach : Averaging base models

    # We begin with this simple approach of averaging base models.  We build a new **class**  to extend scikit-learn with our model and also to laverage encapsulation and code reuse ([inheritance][1])
    #
    #
    #   [1]: https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)

    # **Averaged base models class**

    class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models

        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.models]

            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)

            return self

        # Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)

    # **Averaged base models score**

    # We just average four models here **ENet, GBoost,  KRR and lasso**.  Of course we could easily add more models in the mix.
    averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))

    score = rmsle_cv(averaged_models)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(
        score.mean(), score.std()))

    # Wow ! It seems even the simplest stacking approach really improve the score . This encourages
    # us to go further and explore a less simple stacking approch.

    # ###Less simple Stacking : Adding a Meta-model

    # In this approach, we add a meta-model on averaged base models and use the out-of-folds predictions of these base models to train our meta-model.
    #
    # The procedure, for the training part, may be described as follows:
    #
    #
    # 1. Split the total training set into two disjoint sets (here **train** and .**holdout** )
    #
    # 2. Train several base models on the first part (**train**)
    #
    # 3. Test these base models on the second part (**holdout**)
    #
    # 4. Use the predictions from 3)  (called  out-of-folds predictions) as the inputs, and the correct responses (target variable) as the outputs  to train a higher level learner called **meta-model**.
    #
    # The first three steps are done iteratively . If we take for example a 5-fold stacking , we first split the training data into 5 folds. Then we will do 5 iterations. In each iteration,  we train every base model on 4 folds and predict on the remaining fold (holdout fold).
    #
    # So, we will be sure, after 5 iterations , that the entire data is used to get out-of-folds predictions that we will then use as
    # new feature to train our meta-model in the step 4.
    #
    # For the prediction part , We average the predictions of  all base models on the test data  and used them as **meta-features**  on which, the final prediction is done with the meta-model.
    #

    # ![Faron](http://i.imgur.com/QBuDOjs.jpg)
    #
    # (Image taken from [Faron](https://www.kaggle.com/getting-started/18153#post103381))

    # ![kaz](http://5047-presscdn.pagely.netdna-cdn.com/wp-content/uploads/2017/06/image5.gif)
    #
    # Gif taken from [KazAnova's interview](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)

    # On this gif, the base models are algorithms 0, 1, 2 and the meta-model is algorithm 3. The entire training dataset is
    # A+B (target variable y known) that we can split into train part (A) and holdout part (B). And the test dataset is C.
    #
    # B1 (which is the prediction from the holdout part)  is the new feature used to train the meta-model 3 and C1 (which
    # is the prediction  from the test dataset) is the meta-feature on which the final prediction is done.

    # **Stacking averaged Models Class**

    class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, base_models, meta_model, n_folds=5):
            self.base_models = base_models
            self.meta_model = meta_model
            self.n_folds = n_folds

        # We again fit the data on clones of the original models
        def fit(self, X, y):
            self.base_models_ = [list() for x in self.base_models]
            self.meta_model_ = clone(self.meta_model)
            kfold = KFold(n_splits=self.n_folds,
                          shuffle=True, random_state=156)

            # Train cloned base models then create out-of-fold predictions
            # that are needed to train the cloned meta-model
            out_of_fold_predictions = np.zeros(
                (X.shape[0], len(self.base_models)))
            for i, model in enumerate(self.base_models):
                for train_index, holdout_index in kfold.split(X, y):
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                    out_of_fold_predictions[holdout_index, i] = y_pred

            # Now train the cloned  meta-model using the out-of-fold predictions as new feature
            self.meta_model_.fit(out_of_fold_predictions, y)
            return self

        # Do the predictions of all base models on the test data and use the averaged predictions as
        # meta-features for the final prediction which is done by the meta-model
        def predict(self, X):
            meta_features = np.column_stack([
                np.column_stack([model.predict(X)
                                for model in base_models]).mean(axis=1)
                for base_models in self.base_models_])
            return self.meta_model_.predict(meta_features)

    # **Stacking Averaged models Score**

    # To make the two approaches comparable (by using the same number of models) , we just average **Enet KRR and Gboost**, then we add **lasso as meta-model**.
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)

    score = rmsle_cv(stacked_averaged_models)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(
        score.mean(), score.std()))

    # We get again a better score by adding a meta learner

    # ## Ensembling StackedRegressor, XGBoost and LightGBM

    # We add **XGBoost and LightGBM** to the** StackedRegressor** defined previously.

    # We first define a rmsle evaluation function

    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    # ###Final Training and Prediction

    # **StackedRegressor:**
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    print(rmsle(y_train, stacked_train_pred))

    # **XGBoost:**
    model_xgb.fit(train, y_train)
    xgb_train_pred = model_xgb.predict(train)
    xgb_pred = np.expm1(model_xgb.predict(test))
    print(rmsle(y_train, xgb_train_pred))

    # **LightGBM:**
    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    print(rmsle(y_train, lgb_train_pred))

    '''RMSE on the entire Train data when averaging'''
    print('RMSLE score on train data:')
    score = rmsle(y_train, stacked_train_pred*0.70 +
                   xgb_train_pred*0.15 + lgb_train_pred*0.15)
    metrics = {
        'metrics': [{
            'name': 'rmsle',       # The name of the metric. Visualized as the column name in the runs table.
            'numberValue':  score, # The value of the metric. Must be a numeric value.
            'format': "RAW",       # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
    }
    print(score)
    with open(mlpipeline_metrics_path, 'w') as f:
        json.dump(metrics, f)

    # **Ensemble prediction:**
    ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

    # **Submission**
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv(output_path, index=False)

@dsl.pipeline(
    name='House Prices - Advanced Regression Techniques',
    description='House Prices - Advanced Regression Techniques'
)
def house_price_pipeline(
    epochs=250,
    batch_size=32,
    sha=''
):
    download_data_factory = func_to_container_op(func=download_data,
                                                 base_image='tensorflow/tensorflow:2.4.1',
                                                 packages_to_install=[
                                                     'requests==2.25.0',
                                                 ])
    train_factory = func_to_container_op(func=train,
                                         base_image='tensorflow/tensorflow:2.4.1',
                                         packages_to_install=[
                                             'pathlib2>=2.3.1,<2.4.0',
                                             'requests==2.25.0',
                                             'lightgbm==3.1.1',
                                             'xgboost==1.3.3',
                                             'scikit-learn>=0.24.1',
                                             'pandas>=1.1.5',
                                         ])
    train_data_url = 'https://gist.githubusercontent.com/tcnghia/11d3314fec450511a5d93fd860208575/raw/4a68c3f6c56ced434ce6e634c089b0d1855b084c/train.csv'
    test_data_url = 'https://gist.githubusercontent.com/tcnghia/41b010038e7114690e752649174300cc/raw/728e626f4da41b320c2c548cb4ec7ff799196d00/test.csv'
    train_download_op = download_data_factory(train_data_url)
    train_download_op.set_display_name('Download training data')
    test_download_op = download_data_factory(test_data_url)
    test_download_op.set_display_name('Download test data')
    train_op = train_factory(train_data=train_download_op.output,
                             test_data=test_download_op.output)
    train_op.after(train_download_op)
    train_op.after(test_download_op)

if __name__ == '__main__':
    compiler.Compiler().compile(house_price_pipeline, __file__ + '.tar.gz')
