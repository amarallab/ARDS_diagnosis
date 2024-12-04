"""
Functions to do cross validation by encounters, not records

Author: Felix L. Morales
"""

import string
from nltk import word_tokenize
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    log_loss,
    r2_score,
    average_precision_score
)
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression as logit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from xgboost import XGBClassifier
from statsmodels.stats.stattools import durbin_watson
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll.base import scope
from joblib import Parallel, delayed


def tokenizer_better(text: str) -> list:
    """
    Tokenizes the input text by removing punctuation and digits, and then splitting into words.
    
    Args:
        text (str): The input string to be tokenized.
    Returns:
        list: A list of word tokens extracted from the input text.
    """
    
    punc_list = string.punctuation + '0123456789,'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    #text = text.lower().translate(t)
    tokens = word_tokenize(text)
    
    return tokens


def inner_cv(
    model_type: str,
    df: pd.DataFrame,
    encounters: np.ndarray,
    columns: dict,
    which: str,
    train_index: np.ndarray,
    test_index: np.ndarray,
    score: str = 'auc',
    hyperparams: dict = None
) -> float:
    """
    Perform inner cross-validation for a given model type and dataset.
    
    Parameters:
    model_type (str): The type of model to train. Supported values are "xgboost", "randomforest", "logisticregression", and "decisiontree".
    df (pd.DataFrame): The dataframe containing the data.
    encounters (np.ndarray): Array of encounter IDs.
    columns (dict): Dictionary containing the feature and target column names.
    which (str): Key to access the specific feature and target columns from the columns dictionary.
    train_index (np.ndarray): Array of indices for the training set encounters.
    test_index (np.ndarray): Array of indices for the test set encounters.
    score (str, optional): The scoring metric to use. Supported values are "auc", "aucpr", and "log_loss". Default is "auc".
    hyperparams (dict, optional): Dictionary of hyperparameters for the model.
    
    Returns:
    float: The test score based on the specified scoring metric.
    
    Raises:
    ValueError: If an unsupported model type or scoring scheme is provided.
    """
    
    train_encounters = encounters[train_index]
    test_encounters = encounters[test_index]
    
    train = df['encounter_id'].isin(train_encounters)
    test = df['encounter_id'].isin(test_encounters)
    
    X_train = df.loc[train, columns[which][0]].to_numpy()
    X_test = df.loc[test, columns[which][0]].to_numpy()
    Y_train = df.loc[train, columns[which][1]].to_numpy()
    Y_test = df.loc[test, columns[which][1]].to_numpy()
    
    # Vectorize
    vect = CountVectorizer(
        tokenizer=tokenizer_better,
        ngram_range=(1, 2),
        max_features=200
    )
    
    vect.fit(X_train)
    X_train_vect = vect.transform(X_train).toarray()
    X_test_vect = vect.transform(X_test).toarray()
    
    if model_type.lower() in ["xgboost", "xgb", "gradient boosting", "gradient_boosting", "gradientboosting"]:
        model = XGBClassifier(
            base_score=hyperparams['base_score'],
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            min_child_weight=hyperparams['min_child_weight'],
            max_delta_step=hyperparams['max_delta_step'],
            subsample=hyperparams['subsample'],
            tree_method='hist',
            random_state=0
        )
        
        model.fit(X_train_vect, Y_train)
        
    elif model_type.lower() in ["randomforest", "random_forest", "random forest"]:
        model = RandomForestClassifier(
            n_estimators=hyperparams['n_estimators'],
            criterion=hyperparams['criterion'],
            max_depth=hyperparams['max_depth'],
            min_samples_split=hyperparams['min_samples_split'],
            min_samples_leaf=hyperparams['min_samples_leaf'],
            warm_start=hyperparams['warm_start'],
            class_weight=hyperparams['class_weight'],
            random_state=0
        )
        
        model.fit(X_train_vect, Y_train)
    
    elif model_type.lower() in ["logisticregression", "logistic_regression", "logit", "logistic regression"]:
        model = logit(
            tol=hyperparams['tol'],
            C=hyperparams['C'],
            class_weight=hyperparams['class_weight'],
            max_iter=hyperparams['max_iter'],
            l1_ratio=hyperparams['l1_ratio'],
            warm_start=hyperparams['warm_start'],
            solver='saga',
            penalty='elasticnet',
            random_state=0
        )
        
        model.fit(X_train_vect, Y_train)
        
    elif model_type.lower() in ["decisiontree", "decision_tree", "decision tree",
                                "decisiontrees", "decision_trees", "decision trees"]:
        model = tree.DecisionTreeClassifier(
            criterion=hyperparams['criterion'],
            splitter=hyperparams['splitter'],
            max_features=hyperparams['max_features'],
            min_samples_leaf=hyperparams['min_samples_leaf'],
            min_samples_split=hyperparams['min_samples_split'],
            max_leaf_nodes=hyperparams['max_leaf_nodes'],
            max_depth=hyperparams['max_depth'],
            random_state=0
        )
        
        model.fit(X_train_vect, Y_train)
        
    else:
        raise ValueError("You did not select a supported model")
    
    test_preds = model.predict_proba(X_test_vect)[:, 1]
    
    if score.lower() == 'auc':
        test_score = roc_auc_score(Y_test, test_preds)
    elif score.lower() == 'aucpr':
        test_score = average_precision_score(Y_test, test_preds)
    elif score.lower() == 'log_loss':
        test_score = log_loss(Y_test, test_preds)
    else:
        raise ValueError("Invalid scoring scheme, enter either 'auc' or 'aucpr'")
        
    return test_score



def nested_cv(
    df: pd.DataFrame,
    columns: dict,
    which: str,
    encounters: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    mean_fpr: np.ndarray,
    mean_mpv: np.ndarray,
    model: str = None,
    score: str = 'auc'
) -> tuple:
    """
    Perform nested cross-validation for various machine learning models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    columns : dict
        A dictionary where keys are model names and values are lists of column names for features and target.
    which : str
        The key in the columns dictionary to select the appropriate columns for the model.
    encounters : numpy.ndarray
        Array of encounter IDs.
    train_index : numpy.ndarray
        Indices for the training set.
    test_index : numpy.ndarray
        Indices for the test set.
    mean_fpr : numpy.ndarray
        Mean false positive rate for ROC curve interpolation.
    mean_mpv : numpy.ndarray
        Mean predicted value for calibration curve interpolation.
    model : str, optional
        The machine learning model to use. Supported models are 'xgboost', 'randomforest', 'logisticregression', and 'decisiontree'.
    score : str, optional
        The scoring metric to use. Options are 'auc', 'aucpr', and 'log_loss'. Default is 'auc'.
        
    Returns:
    --------
    tuple:
        - test_score (float): The score of the model on the test set based on the specified scoring metric.
        - test_preds (list): List of predicted probabilities for the test set.
        - interp_tpr (numpy.ndarray): Interpolated true positive rate for ROC curve.
        - interp_fop (numpy.ndarray): Interpolated fraction of positives for calibration curve.
        - r_squared (float): R-squared value for the calibration curve.
        - dw (float): Durbin-Watson statistic for the calibration curve.
        - importances (list): List of dictionaries containing feature importances.
        
    Raises:
    -------
    ValueError
        If an unsupported model or invalid scoring scheme is provided.
    """
    
    # Split data into train and test based on "encounter indices"
    train_encounters = encounters[train_index]
    test_encounters = encounters[test_index]
    
    train = df['encounter_id'].isin(train_encounters)
    test = df['encounter_id'].isin(test_encounters)
    
    training_data = df.loc[train]
    
    X_train = df.loc[train, columns[which][0]].to_numpy()
    X_test = df.loc[test, columns[which][0]].to_numpy()
    Y_train = df.loc[train, columns[which][1]].to_numpy()
    Y_test = df.loc[test, columns[which][1]].to_numpy()
    
    # Vectorize
    vect = CountVectorizer(
        tokenizer=tokenizer_better,
        ngram_range=(1, 2),
        max_features=200
    )
    
    vect.fit(X_train)
    X_train_vect = vect.transform(X_train).toarray()
    X_test_vect = vect.transform(X_test).toarray()
    features = {value: key for key, value in vect.vocabulary_.items()}
    
    if model.lower() in ["xgboost", "xgb", "gradient boosting", "gradient_boosting", "gradientboosting"]:
        # Optimize hyperparameters on training set
        XG_param_grid = {
            'base_score': hp.uniform('base_score', 0.0, 1.0),
            'n_estimators': scope.int(hp.quniform("n_estimators", 10, 10000, 10)),
            'max_depth': scope.int(hp.quniform("max_depth", 10, 1000, 10)),
            'learning_rate': hp.uniform('learning_rate', 0.0, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 10.0),
            'min_child_weight': hp.uniform('min_child_weight', 0.0, 100.0),
            'max_delta_step': hp.uniform("max_delta_step", 0.0, 100.0),
            'subsample': hp.uniform("subsample", 0.001, 1.0)
        }
    
        def objective(XG_param_grid):
            cv = KFold(n_splits=3)
            logloss = Parallel(n_jobs=3)(delayed(inner_cv)(
                model,
                training_data,
                train_encounters,
                columns,
                which,
                train_index,
                test_index,
                score='log_loss',
                hyperparams=XG_param_grid
            ) for train_index, test_index in cv.split(train_encounters))
    
            mean_logloss = np.mean(logloss)
            var_logloss = np.var(logloss, ddof=1)

            return {'loss': mean_logloss, 'loss_variance': var_logloss, 'status': STATUS_OK}
    
        hyperparams = fmin(
            fn=objective,
            space=XG_param_grid,
            algo=tpe.suggest,
            max_evals=160,
            trials=Trials(),
            early_stop_fn=no_progress_loss(40)
        )
        
        hyperparams['n_estimators'] = int(hyperparams['n_estimators'])
        hyperparams['max_depth'] = int(hyperparams['max_depth'])
    
        modelo = XGBClassifier(**hyperparams, tree_method='hist', random_state=0)
        modelo.fit(X_train_vect, Y_train)
        raw_coeffs = modelo.feature_importances_
        
    elif model.lower() in ["randomforest", "random_forest", "random forest"]:
        criterion = ['entropy', 'gini', 'log_loss']
        warm_start = [True, False]
        class_weight = ['balanced', 'balanced_subsample', None]

        RF_param_grid = {
            'n_estimators': scope.int(hp.quniform("n_estimators", 100, 5000, 10)),
            'criterion': hp.choice('criterion', criterion),
            'max_depth': scope.int(hp.quniform("max_depth", 100, 5000, 10)),
            'min_samples_split': scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            'min_samples_leaf': scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
            'warm_start': hp.choice('warm_start', warm_start),
            'class_weight': hp.choice('class_weight', class_weight)
        }
        
        def objective(RF_param_grid):
            cv = KFold(n_splits=3)
            logloss = Parallel(n_jobs=3)(delayed(inner_cv)(
                model,
                training_data,
                train_encounters,
                columns,
                which,
                train_index,
                test_index,
                score='log_loss',
                hyperparams=RF_param_grid
            ) for train_index, test_index in cv.split(train_encounters))
    
            mean_logloss = np.mean(logloss)
            var_logloss = np.var(logloss, ddof=1)
    
            return {'loss': mean_logloss, 'loss_variance': var_logloss, 'status': STATUS_OK}
        
        hyperparams = fmin(
            fn=objective,
            space=RF_param_grid,
            algo=tpe.suggest,
            max_evals=200,
            trials=Trials(),
            early_stop_fn=no_progress_loss(50)
        )
        
        hyperparams['n_estimators'] = int(hyperparams['n_estimators'])
        hyperparams['criterion'] = criterion[hyperparams['criterion']]
        hyperparams['max_depth'] = int(hyperparams['max_depth'])
        hyperparams['min_samples_split'] = int(hyperparams['min_samples_split'])
        hyperparams['min_samples_leaf'] = int(hyperparams['min_samples_leaf'])
        hyperparams['warm_start'] = warm_start[hyperparams['warm_start']]
        hyperparams['class_weight'] = class_weight[hyperparams['class_weight']]
        
        modelo = RandomForestClassifier(**hyperparams, random_state=0)
        modelo.fit(X_train_vect, Y_train)
        raw_coeffs = modelo.feature_importances_
        
    elif model.lower() in ["logisticregression", "logistic_regression", "logit", "logistic regression"]:
        class_weight = ['balanced', None]
        warm_start = [True, False]

        LR_param_grid = {
            'tol': hp.uniform('tol', 1e-6, 1e-1),
            'C': hp.uniform('C', 1e-6, 1e6),
            'class_weight': hp.choice('class_weight', class_weight),
            'max_iter': scope.int(hp.quniform("max_iter", 100, 4000, 1)),
            'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
            'warm_start': hp.choice('warm_start', warm_start)
        }
        
        def objective(LR_param_grid):
            cv = KFold(n_splits=3) 
            logloss = Parallel(n_jobs=3)(delayed(inner_cv)(
                model,
                training_data,
                train_encounters,
                columns,
                which,
                train_index,
                test_index,
                score='log_loss',
                hyperparams=LR_param_grid
            ) for train_index, test_index in cv.split(train_encounters))
    
            mean_logloss = np.mean(logloss)
            var_logloss = np.var(logloss, ddof=1)
    
            return {'loss': mean_logloss, 'loss_variance': var_logloss, 'status': STATUS_OK}
        
        hyperparams = fmin(
            fn=objective,
            space=LR_param_grid,
            algo=tpe.suggest,
            max_evals=140,
            trials=Trials(),
            early_stop_fn=no_progress_loss(35)
        )
            
        hyperparams['class_weight'] = class_weight[hyperparams['class_weight']]
        hyperparams['max_iter'] = int(hyperparams['max_iter'])
        hyperparams['warm_start'] = warm_start[hyperparams['warm_start']]
        
        modelo = logit(**hyperparams, solver='saga', penalty='elasticnet', random_state=0)
        modelo.fit(X_train_vect, Y_train)
        raw_coeffs = modelo.coef_[0]
        
    elif model.lower() in ["decisiontree", "decision_tree", "decision tree",
                           "decisiontrees", "decision_trees", "decision trees"]:
        criterion = ['entropy', 'gini', 'log_loss']
        splitter = ['best', 'random']
        max_features = ['sqrt', 'log2', None]

        DT_param_grid = {
            'criterion': hp.choice("criterion", criterion),
            'splitter': hp.choice("splitter", splitter),
            'max_depth': scope.int(hp.quniform("max_depth", 1, 1000, 1)),
            'min_samples_split': scope.int(hp.quniform("min_samples_split", 2, 200, 1)),
            'min_samples_leaf': scope.int(hp.quniform("min_samples_leaf", 1, 100, 1)),
            'max_features': hp.choice("max_features", max_features),
            'max_leaf_nodes': scope.int(hp.quniform("max_leaf_nodes", 2, 1000, 1))
        }
        
        def objective(DT_param_grid):
            cv = KFold(n_splits=3)
            logloss = Parallel(n_jobs=3)(delayed(inner_cv)(
                model,
                training_data,
                train_encounters,
                columns,
                which,
                train_index,
                test_index,
                score='log_loss',
                hyperparams=DT_param_grid
            ) for train_index, test_index in cv.split(train_encounters))
    
            mean_logloss = np.mean(logloss)
            var_logloss = np.var(logloss, ddof=1)
    
            return {'loss': mean_logloss, 'loss_variance': var_logloss, 'status': STATUS_OK}
        
        hyperparams = fmin(
            fn=objective,
            space=DT_param_grid,
            algo=tpe.suggest,
            max_evals=200,
            trials=Trials(),
            early_stop_fn=no_progress_loss(50)
        )
            
        hyperparams['criterion'] = criterion[hyperparams['criterion']]
        hyperparams['splitter'] = splitter[hyperparams['splitter']]
        hyperparams['max_depth'] = int(hyperparams['max_depth'])
        hyperparams['max_features'] = max_features[hyperparams['max_features']]
        hyperparams['min_samples_split'] = int(hyperparams['min_samples_split'])
        hyperparams['min_samples_leaf'] = int(hyperparams['min_samples_leaf'])
        hyperparams['max_leaf_nodes'] = int(hyperparams['max_leaf_nodes'])
        
        modelo = tree.DecisionTreeClassifier(**hyperparams, random_state=0)
        modelo.fit(X_train_vect, Y_train)
        raw_coeffs = modelo.feature_importances_
        
    else:
        raise ValueError("You did not select a supported model. See docstring for supported ML models.")
        
    # Predictions
    test_preds = modelo.predict_proba(X_test_vect)[:, 1]
    
    # Scoring
    if score == 'auc':
        test_score = roc_auc_score(Y_test, test_preds)
    elif score == 'aucpr':
        test_score = average_precision_score(Y_test, test_preds)
    elif score == 'log_loss':
        test_score = log_loss(Y_test, test_preds)
    else:
        raise ValueError("Invalid scoring scheme, enter either 'auc' or 'brier'")
    
    # # ROC curve
    # fpr_test, tpr_test, _ = roc_curve(Y_test, test_preds)
    # interp_tpr = np.interp(mean_fpr, fpr_test, tpr_test)
    # interp_tpr[0] = 0.0
    
    # AUCPR curve
    precision_test, recall_test, _ = precision_recall_curve(Y_test, test_preds)
    interp_tpr = np.interp(mean_fpr, recall_test, precision_test)
    interp_tpr[0] = 1.0
    
    # Calibration curve
    fop, mpv = calibration_curve(Y_test, test_preds, n_bins=20)
    interp_fop = np.interp(mean_mpv, mpv, fop)
    r_squared = r2_score(interp_fop, mean_mpv)
    dw = durbin_watson(interp_fop - mean_mpv)
    
    # Importances
    importances = []
    for i, coeff in enumerate(raw_coeffs):
        importances.append({'feature': features[i], 'importance': coeff})
        
    return test_score, list(test_preds), interp_tpr, interp_fop, r_squared, dw, importances
    

def custom_cv_not_train(
    model: 'sklearn.base.BaseEstimator',
    df: pd.DataFrame,
    columns: dict,
    which: str,
    encounters: list,
    vectorizer: CountVectorizer,
    test_index: list,
    mean_fpr: np.ndarray,
    mean_mpv: np.ndarray,
    score: str = 'auc'
) -> tuple:
    """
    Perform custom cross-validation without training the model.
    
    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to evaluate.
    df (pandas.DataFrame): The dataframe containing the data.
    columns (dict): A dictionary where keys are column names and values are lists of feature and target column names.
    which (str): The key in the columns dictionary to use for features and target.
    encounters (list): A list of encounter identifiers.
    vectorizer (sklearn.feature_extraction.text.CountVectorizer): The vectorizer to transform text data.
    test_index (list): The indices of the test encounters.
    mean_fpr (numpy.ndarray): The mean false positive rate for ROC curve interpolation.
    mean_mpv (numpy.ndarray): The mean predicted value for calibration curve interpolation.
    score (str, optional): The scoring metric to use ('auc', 'aucpr', or 'log_loss'). Default is 'auc'.
    
    Returns:
    tuple: A tuple containing:
        - test_score (float): The score of the model on the test set.
        - test_preds (list): The predicted probabilities for the test set.
        - interp_tpr (numpy.ndarray): The interpolated true positive rates for the ROC curve.
        - interp_fop (numpy.ndarray): The interpolated fraction of positives for the calibration curve.
        - importances (list): A list of dictionaries containing feature importances.
        
    Raises:
    ValueError: If an invalid scoring scheme is provided.
    """
    
    test_encounters = encounters[test_index]
    
    try:
        test = df['icu_id'].isin(test_encounters)
    except KeyError:
        try:
            test = df['encounter_id'].isin(test_encounters)
        except KeyError:
            test = df['_id'].isin(test_encounters)
    
    X_test = df.loc[test, columns[which][0]].to_numpy()
    Y_test = df.loc[test, columns[which][1]].to_numpy()
    
    X_test_vect = vectorizer.transform(X_test).toarray()
    features = {value: key for key, value in vectorizer.vocabulary_.items()}
    
    # Predictions
    test_preds = model.predict_proba(X_test_vect)[:, 1]
    
    # Scoring
    if score == 'auc':
        test_score = roc_auc_score(Y_test, test_preds)
    elif score == 'aucpr':
        test_score = average_precision_score(Y_test, test_preds)
    elif score == 'log_loss':
        test_score = log_loss(Y_test, test_preds)
    else:
        raise ValueError("Invalid scoring scheme, enter either 'auc' or 'brier'")
    
    # ROC curve
    fpr_test, tpr_test, _ = roc_curve(Y_test, test_preds)
    interp_tpr = np.interp(mean_fpr, fpr_test, tpr_test)
    interp_tpr[0] = 0.0
    
    # # AUCPR curve
    # precision_test, recall_test, _ = precision_recall_curve(Y_test, test_preds)
    # interp_tpr = np.interp(mean_fpr, recall_test, precision_test)
    # interp_tpr[0] = 1.0
    
    # Calibration curve
    fop, mpv = calibration_curve(Y_test, test_preds, n_bins=10)
    interp_fop = np.interp(mean_mpv, mpv, fop)
    
    # Importances
    importances = []
    try:
        model.feature_importances_
    except AttributeError:
        raw_coeffs = model.coef_[0]
        for i, coeff in enumerate(raw_coeffs):
            importances.append({'feature': features[i], 'importance': coeff})
    else:
        raw_coeffs = model.feature_importances_
        for i, coeff in enumerate(raw_coeffs):
            importances.append({'feature': features[i], 'importance': coeff})
        
    return test_score, list(test_preds), interp_tpr, interp_fop, importances
