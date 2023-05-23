import shap
import pickle
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import repeat
from fastnode2vec import Graph, Node2Vec
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import csv
import math
import numpy as np
import pandas as pd
import numpy.ma as ma
import networkx as nx
import matplotlib.pyplot as plt

from scipy import stats
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
import functools as ft

from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#from sklearn.pipeline import Pipeline

from matplotlib.pyplot import figure

from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numba as nb

import os
import sys
import glob 

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

@nb.jit(parallel=True)
def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

#helper functions
def str_to_time(date_time_str):
    if isinstance(date_time_str,str) is False and math.isnan(date_time_str):
        date_time_obj = None
    else:    
        date_time_obj = datetime.strptime(date_time_str, "%m/%d/%Y %H:%M")
    return date_time_obj
def time_diff_transform(variable_array, unit_rate):
    output_variable_array = []
    for row in variable_array:
        tmp_row = []
        for idx in range(1, len(row)):
            if row[idx-1] is None or row[idx] is None:
                tmp_row.append(math.nan)
            else:
                diff = (row[idx]-row[idx-1]).total_seconds() / unit_rate
                tmp_row.append(diff)
        output_variable_array.append(tmp_row)
    return output_variable_array
def fill_missing_ASA(anes_list, prior_list):
    for idx in range(len(prior_list)):
        target_value = prior_list[idx]
        if isinstance(target_value,str) is False and math.isnan(target_value):
            prior_list[idx] = anes_list[idx]
    return prior_list
def fill_missing_cont_var(input_list, mode = "mean"):
    if (mode == "mean"):
        result_list = np.where(np.isnan(input_list), ma.array(input_list, mask=np.isnan(input_list)).mean(axis=0), input_list)
    else:
        result_list = np.where(np.isnan(input_list), stats.mode(ma.array(input_list, mask=np.isnan(input_list)),axis=0)[0], input_list) 
    return result_list
#icu

def agg_sta_featues(x):
  return [np.mean(x),np.median(x),iqr_calculator(x)]
def concat_id_icd(encounter_id, icd_list,poa_status):
    return pd.DataFrame({'encounter_id': list(repeat(encounter_id,len(np.array(icd_list)))),
                         'icd10_transformed':icd_list,
                         'POA':list(repeat(poa_status,len(np.array(icd_list))))})

#helper function
def str_to_time(date_time_str):
    if isinstance(date_time_str,str) is False and math.isnan(date_time_str):
        date_time_obj = None
    else:    
        date_time_obj = datetime.strptime(date_time_str, "%m/%d/%Y %H:%M")
    return date_time_obj
def model_evalution_binary(model, X_train, X_test, y_train, y_test):
    reg = model.fit(X_train, y_train)
    y_pred_test = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    
    y_pred_proba_test = reg.predict_proba(X_test)[:,1]
    y_pred_proba_train = reg.predict_proba(X_train)[:,1]
    
    auc_test = np.round(roc_auc_score(y_test, y_pred_proba_test),5)
    f1_test = np.round(f1_score(y_test, y_pred_test),5)
    precision_test = np.round(precision_score(y_test, y_pred_test),5)
    recall_test = np.round(recall_score(y_test, y_pred_test),5)
    accuracy_test = np.round(accuracy_score(y_test, y_pred_test),5)

        
    auc_train = np.round(roc_auc_score(y_train, y_pred_proba_train),5)
    f1_train = np.round(f1_score(y_train, y_pred_train),5)
    precision_train = np.round(precision_score(y_train, y_pred_train),5)
    recall_train = np.round(recall_score(y_train, y_pred_train),5)
    accuracy_train = np.round(accuracy_score(y_train, y_pred_train),5)
    
    #print(y_pred, y_test)
    return auc_test,f1_test,precision_test,recall_test,accuracy_test, auc_train,f1_train,precision_train,recall_train,accuracy_train


def experiment_baseline(X_train, X_test, y_train, y_test, seeds=[2023,2024], xgb_colsample_bytree=0.8764572683079066, xgb_learning_rate=0.033046819240715494,xgb_max_depth=7,xgb_n_estimators=460,xgb_subsample=1.0,mlp_early_stopping=False,mlp_learning_rate='adaptive',mlp_alpha=0.1009782709585233,lr_max_iter=500,lr_solver='newton-cg',lr_C=49.99999999999999,lr_class_weight='balanced'):

    index_train = y_train[['ENCRYPTED_HOSP_ENCOUNTER']]
    
    X_train = pd.merge(index_train,\
                       X_train,\
                       on='ENCRYPTED_HOSP_ENCOUNTER',\
                       how='left').\
                drop(columns=['ENCRYPTED_HOSP_ENCOUNTER'])
    
    y_train = y_train.drop(columns=['ENCRYPTED_HOSP_ENCOUNTER'])
    
    index_test = y_test[['ENCRYPTED_HOSP_ENCOUNTER']]
    
    X_test = pd.merge(index_test,\
                       X_test,\
                       on='ENCRYPTED_HOSP_ENCOUNTER',\
                       how='left').\
                drop(columns=['ENCRYPTED_HOSP_ENCOUNTER'])
    
    y_test = y_test.drop(columns=['ENCRYPTED_HOSP_ENCOUNTER'])
    
    
    rf_auc_test = []
    rf_auc_train=[]
    xgb_auc_test=[]
    xgb_auc_train=[]
    mlp_auc_test=[]
    mlp_auc_train=[]
    log_auc_test=[]
    log_auc_train=[]

    rf_f1_test = []
    rf_f1_train=[]
    xgb_f1_test=[]
    xgb_f1_train=[]
    mlp_f1_test=[]
    mlp_f1_train=[]
    log_f1_test=[]
    log_f1_train=[]

    rf_precision_test = []
    rf_precision_train=[]
    xgb_precision_test=[]
    xgb_precision_train=[]
    mlp_precision_test=[]
    mlp_precision_train=[]
    log_precision_test=[]
    log_precision_train=[]

    rf_recall_test = []
    rf_recall_train=[]
    xgb_recall_test=[]
    xgb_recall_train=[]
    mlp_recall_test=[]
    mlp_recall_train=[]
    log_recall_test=[]
    log_recall_train=[]

    rf_accuracy_test = []
    rf_accuracy_train=[]
    xgb_accuracy_test=[]
    xgb_accuracy_train=[]
    mlp_accuracy_test=[]
    mlp_accuracy_train=[]
    log_accuracy_test=[]
    log_accuracy_train=[]
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_train.values)
    X_train = imp.transform(X_train.values)
    X_test = imp.transform(X_test.values)

    my_scaler = MinMaxScaler()
    my_scaler.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)
    
    for i in tqdm(seeds):

        np.random.seed(i)

        

        xgb_model = XGBClassifier(colsample_bytree=xgb_colsample_bytree,\
                                  learning_rate=xgb_learning_rate,\
                                  max_depth=xgb_max_depth,\
                                  n_estimators = xgb_n_estimators,\
                                  subsample = xgb_subsample)
        
        mlp_model = MLPClassifier(learning_rate=mlp_learning_rate,\
                                  early_stopping = mlp_early_stopping,\
                                  alpha = mlp_alpha)
        
        log_model = LogisticRegression(max_iter=lr_max_iter,\
                                       solver=lr_solver,\
                                       C = lr_C,\
                                       class_weight = lr_class_weight)

        xgb_auc_test_tmp,xgb_f1_test_tmp,xgb_precision_test_tmp,xgb_recall_test_tmp,xgb_accuracy_test_tmp, xgb_auc_train_tmp,xgb_f1_train_tmp,xgb_precision_train_tmp,xgb_recall_train_tmp,xgb_accuracy_train_tmp = model_evalution_binary(xgb_model, X_train, X_test, y_train, y_test)
        mlp_auc_test_tmp,mlp_f1_test_tmp,mlp_precision_test_tmp,mlp_recall_test_tmp,mlp_accuracy_test_tmp, mlp_auc_train_tmp,mlp_f1_train_tmp,mlp_precision_train_tmp,mlp_recall_train_tmp,mlp_accuracy_train_tmp = model_evalution_binary(mlp_model, X_train, X_test, y_train, y_test)
        log_auc_test_tmp,log_f1_test_tmp,log_precision_test_tmp,log_recall_test_tmp,log_accuracy_test_tmp, log_auc_train_tmp,log_f1_train_tmp,log_precision_train_tmp,log_recall_train_tmp,log_accuracy_train_tmp = model_evalution_binary(log_model, X_train, X_test, y_train, y_test)


        xgb_auc_test.append(xgb_auc_test_tmp)
        xgb_auc_train.append(xgb_auc_train_tmp)
        mlp_auc_test.append(mlp_auc_test_tmp)
        mlp_auc_train.append(mlp_auc_train_tmp)
        log_auc_test.append(log_auc_test_tmp)
        log_auc_train.append(log_auc_train_tmp)


        xgb_f1_test.append(xgb_f1_test_tmp)
        xgb_f1_train.append(xgb_f1_train_tmp)
        mlp_f1_test.append(mlp_f1_test_tmp)
        mlp_f1_train.append(mlp_f1_train_tmp)
        log_f1_test.append(log_f1_test_tmp)
        log_f1_train.append(log_f1_train_tmp)


        xgb_precision_test.append(xgb_precision_test_tmp)
        xgb_precision_train.append(xgb_precision_train_tmp)
        mlp_precision_test.append(mlp_precision_test_tmp)
        mlp_precision_train.append(mlp_precision_train_tmp)
        log_precision_test.append(log_precision_test_tmp)
        log_precision_train.append(log_precision_train_tmp)


        xgb_recall_test.append(xgb_recall_test_tmp)
        xgb_recall_train.append(xgb_recall_train_tmp)
        mlp_recall_test.append(mlp_recall_test_tmp)
        mlp_recall_train.append(mlp_recall_train_tmp)
        log_recall_test.append(log_recall_test_tmp)
        log_recall_train.append(log_recall_train_tmp)


        xgb_accuracy_test.append(xgb_accuracy_test_tmp)
        xgb_accuracy_train.append(xgb_accuracy_train_tmp)
        mlp_accuracy_test.append(mlp_accuracy_test_tmp)
        mlp_accuracy_train.append(mlp_accuracy_train_tmp)
        log_accuracy_test.append(log_accuracy_test_tmp)
        log_accuracy_train.append(log_accuracy_train_tmp)
    print("====================================================")
    print("A. Scores on training set")
    print("avg XGBoost classifer: auc/ ", np.round(np.mean(xgb_auc_train),4)," F1: ",np.round(np.mean(xgb_f1_train),4)," precision: ",np.round(np.mean(xgb_precision_train),4)," recall: ",np.round(np.mean(xgb_recall_train),4)," accuracy: ",np.round(np.mean(xgb_accuracy_train),4))
    print("std XGBoost classifer: auc/ ", np.round(np.std(xgb_auc_train),4)," F1: ",np.round(np.std(xgb_f1_train),4)," precision: ",np.round(np.std(xgb_precision_train),4)," recall: ",np.round(np.std(xgb_recall_train),4)," accuracy: ",np.round(np.std(xgb_accuracy_train),4))

    print("avg MLP classifer: auc/ ", np.round(np.mean(mlp_auc_train),4)," F1: ",np.round(np.mean(mlp_f1_train),4)," precision: ",np.round(np.mean(mlp_precision_train),4)," recall: ",np.round(np.mean(mlp_recall_train),4)," accuracy: ",np.round(np.mean(mlp_accuracy_train),4))
    print("std MLP classifer: auc/ ", np.round(np.std(mlp_auc_train),4)," F1: ",np.round(np.std(mlp_f1_train),4)," precision: ",np.round(np.std(mlp_precision_train),4)," recall: ",np.round(np.std(mlp_recall_train),4)," accuracy: ",np.round(np.std(mlp_accuracy_train),4))

    print("avg logistic classifer: auc/ ", np.round(np.mean(log_auc_train),4)," F1: ",np.round(np.mean(log_f1_train),4)," precision: ",np.round(np.mean(log_precision_train),4)," recall: ",np.round(np.mean(log_recall_train),4)," accuracy: ",np.round(np.mean(log_accuracy_train),4))
    print("std logistic classifer: auc/ ", np.round(np.std(log_auc_train),4)," F1: ",np.round(np.std(log_f1_train),4)," precision: ",np.round(np.std(log_precision_train),4)," recall: ",np.round(np.std(log_recall_train),4)," accuracy: ",np.round(np.std(log_accuracy_train),4))

    print("****************************************************\n")
    print("B. Scores on test set")
    print("avg XGBoost classifer: auc/ ", np.round(np.mean(xgb_auc_test),4)," F1: ",np.round(np.mean(xgb_f1_test),4)," precision: ",np.round(np.mean(xgb_precision_test),4)," recall: ",np.round(np.mean(xgb_recall_test),4)," accuracy: ",np.round(np.mean(xgb_accuracy_test),4))
    print("std XGBoost classifer: auc/ ", np.round(np.std(xgb_auc_test),4)," F1: ",np.round(np.std(xgb_f1_test),4)," precision: ",np.round(np.std(xgb_precision_test),4)," recall: ",np.round(np.std(xgb_recall_test),4)," accuracy: ",np.round(np.std(xgb_accuracy_test),4))

    print("avg MLP classifer: auc/ ", np.round(np.mean(mlp_auc_test),4)," F1: ",np.round(np.mean(mlp_f1_test),4)," precision: ",np.round(np.mean(mlp_precision_test),4)," recall: ",np.round(np.mean(mlp_recall_test),4)," accuracy: ",np.round(np.mean(mlp_accuracy_test),4))
    print("std MLP classifer: auc/ ", np.round(np.std(mlp_auc_test),4)," F1: ",np.round(np.std(mlp_f1_test),4)," precision: ",np.round(np.std(mlp_precision_test),4)," recall: ",np.round(np.std(mlp_recall_test),4)," accuracy: ",np.round(np.std(mlp_accuracy_test),4))

    print("avg logistic classifer: auc/ ", np.round(np.mean(log_auc_test),4)," F1: ",np.round(np.mean(log_f1_test),4)," precision: ",np.round(np.mean(log_precision_test),4)," recall: ",np.round(np.mean(log_recall_test),4)," accuracy: ",np.round(np.mean(log_accuracy_test),4))
    print("std logistic classifer: auc/ ", np.round(np.std(log_auc_test),4)," F1: ",np.round(np.std(log_f1_test),4)," precision: ",np.round(np.std(log_precision_test),4)," recall: ",np.round(np.std(log_recall_test),4)," accuracy: ",np.round(np.std(log_accuracy_test),4))

    print("====================================================\n")




def my_split(X, y, t_size = 1/10, v_size = 2/9, rand_s = 0):
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=t_size, random_state=rand_s)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=v_size, random_state=rand_s)
    
    return X_train_all, X_train, X_val, X_test, y_train_all, y_train, y_val, y_test

def get_graph_metrics_list(provider_list, graph_metric_dict):
    result_list = []
    for relevant_provider in provider_list:
        result_list.append(graph_metric_dict[relevant_provider])
    return np.array(result_list)

def my_find_community(k,com):
    com_label = np.nan
    g = 1
    for i in com:
        i = np.array(list(i))
        if np.isin(k,i):
            com_label = 'community'+str(g)
            break
        g=g+1
    return com_label
def iqr_calculator(x):
  q75, q25 = np.percentile(x, [75 ,25])
  iqr = q75 - q25
  return iqr
def fill_empty_list(input_list):
  if len(input_list) < 1:
    return [0]
  else: return input_list
def agg_sta_featues_icu(x):
  return [np.mean(x),np.median(x),iqr_calculator(x),np.min(x),np.max(x)]

def concat_id_icd(hosp_encounter_id, icd_list,poa_status):
    return pd.DataFrame({'ENCRYPTED_HOSP_ENCOUNTER': list(repeat(hosp_encounter_id,len(np.array(icd_list)))),
                         'icd10_transformed':icd_list,
                         'POA':list(repeat(poa_status,len(np.array(icd_list))))})