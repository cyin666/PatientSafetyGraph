import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
from utility.preprocessing import *
import time

print('Preprocessing ...')
start_time = time.time()

print('reading adjacency matrix csv files ...')
enc_pvd = pd.read_csv('adjacency_EncounterProvider.csv',index_col=0)
enc_pvd.index = enc_pvd.index.set_names(['ENCRYPTED_HOSP_ENCOUNTER'])
enc_pvd = enc_pvd.reset_index()

enc_icd = pd.read_csv('adjacency_EncounterICD.csv',index_col=0)
enc_icd.index = enc_icd.index.set_names(['ENCRYPTED_HOSP_ENCOUNTER'])
enc_icd = enc_icd.reset_index()

enc_icu = pd.read_csv('adjacency_EncounterICU.csv',index_col=0)
enc_icu.index = enc_icu.index.set_names(['ENCRYPTED_HOSP_ENCOUNTER'])
enc_icu = enc_icu.reset_index()

enc_label = pd.read_csv('hospenc_label.csv',index_col=0)

print('Making row orders consistent')
#pat = enc_date['patient_ir_id'].unique()

enc_pvd = pd.merge(enc_label[['ENCRYPTED_HOSP_ENCOUNTER']],enc_pvd,how='left',on='ENCRYPTED_HOSP_ENCOUNTER')#
enc_pvd = enc_pvd.set_index('ENCRYPTED_HOSP_ENCOUNTER') 

enc_icd = pd.merge(enc_label[['ENCRYPTED_HOSP_ENCOUNTER']],enc_icd,how='left',on='ENCRYPTED_HOSP_ENCOUNTER')
enc_icd = enc_icd.set_index('ENCRYPTED_HOSP_ENCOUNTER') 

enc_icu = pd.merge(enc_label[['ENCRYPTED_HOSP_ENCOUNTER']],enc_icu,how='left',on='ENCRYPTED_HOSP_ENCOUNTER')
enc_icu = enc_icu.set_index('ENCRYPTED_HOSP_ENCOUNTER') 


n_enc, n_pvd = enc_pvd.shape
_, n_icd = enc_icd.shape
_, n_icu = enc_icu.shape


print('constructing tensors of adjacency matrix ...')
'''for pandas > 0.25'''
enc_pvd_adj = torch.FloatTensor(enc_pvd.values).to_sparse().cuda()

enc_icd_adj = sparse_to_tensor(enc_icd.astype(pd.SparseDtype("float", np.nan)).sparse.to_coo()).cuda()

enc_icu_adj = torch.FloatTensor(enc_icu.astype(bool).astype(np.uint8).values).to_sparse().cuda()

print('processing node features ...')
'''
enc_feat = torch.FloatTensor(pd.read_csv('node_feat_encounter_baseline.csv',index_col=1).values).cuda()
icd_feat = torch.FloatTensor(pd.read_csv('node_feat_icd.csv',index_col=0).values).cuda()
icu_feat = torch.FloatTensor(pd.read_csv('node_feat_icu.csv',index_col=0).values).cuda()
pvd_feat = torch.FloatTensor(pd.read_csv('node_feat_provider.csv',index_col=0).values).cuda()
'''

enc_feat = sparse_to_tensor(sp.identity(n_enc)).cuda()
pvd_feat = sparse_to_tensor(sp.identity(n_pvd)).cuda()
icd_feat = sparse_to_tensor(sp.identity(n_icd)).cuda()
icu_feat = sparse_to_tensor(sp.identity(n_icu)).cuda()


print("number of encounters: ", n_enc)
print("number of providers: ", n_pvd)
print("number of icd: ", n_icd)
print("number of icu: ", n_icu)


f_enc = enc_feat.shape[1]
f_pvd = pvd_feat.shape[1]
f_icd = icd_feat.shape[1]
f_icu = icu_feat.shape[1]


print("number of encounter features: ", f_enc)
print("number of provider features: ", f_pvd)
print("number of icd features: ", f_icd)
print("number of icu features: ", f_icu)


print('encounter feature matrix shape:',enc_feat.shape)
print('provider feature matrix shape:',pvd_feat.shape)
print('icd feature matrix shape:',icd_feat.shape)
print('icu feature matrix shape:',icu_feat.shape)

'''
adj_mats={
    (0,1): [enc_icu_adj],
    (0,2): [enc_icd_adj],
    (0,3): [enc_pvd_adj],
}


fea_mats={0: enc_feat, 1: icu_feat, 2: icd_feat, 3:  pvd_feat}

'''
fea_num_identity = {0: n_enc, 1: n_icu, 2: n_icd, 3:  n_pvd}
fea_num_node = {0: f_enc, 1: f_icu, 2: f_icd, 3:  f_pvd}

y_label = torch.FloatTensor(enc_label['PLOS'].values).cuda()

# train/validation/testing spiliting by patients

print('splitting train validation test sets ...')

enc_train = pd.read_csv('multimod_train_index.csv').values
enc_val = pd.read_csv('multimod_val_index.csv').values
enc_test = pd.read_csv('multimod_test_index.csv').values


enc_train_index = np.where(np.isin(enc_label['ENCRYPTED_HOSP_ENCOUNTER'],enc_train))
enc_val_index = np.where(np.isin(enc_label['ENCRYPTED_HOSP_ENCOUNTER'],enc_val))
enc_test_index = np.where(np.isin(enc_label['ENCRYPTED_HOSP_ENCOUNTER'],enc_test))

enc_label = enc_label.set_index('ENCRYPTED_HOSP_ENCOUNTER')

# train/validation/testing labels spiliting by patients

train_fea_mats={0: to_dense(enc_feat)[enc_train_index], 1: to_dense(icu_feat), 2: to_dense(icd_feat), 3: to_dense(pvd_feat)}
val_fea_mats={0: to_dense(enc_feat)[enc_val_index], 1: to_dense(icu_feat), 2: to_dense(icd_feat), 3: to_dense(pvd_feat)}
test_fea_mats={0: to_dense(enc_feat)[enc_test_index], 1: to_dense(icu_feat), 2: to_dense(icd_feat), 3: to_dense(pvd_feat)}




train_adj_mats={
          (0,1): [to_dense(enc_icu_adj)[enc_train_index]],
          (0,2): [to_dense(enc_icd_adj)[enc_train_index]],
          (0,3): [to_dense(enc_pvd_adj)[enc_train_index]],
         }

val_adj_mats={
          (0,1): [to_dense(enc_icu_adj)[enc_val_index]],
          (0,2): [to_dense(enc_icd_adj)[enc_val_index]],
          (0,3): [to_dense(enc_pvd_adj)[enc_val_index]],
         }

test_adj_mats={
          (0,1): [to_dense(enc_icu_adj)[enc_test_index]],
          (0,2): [to_dense(enc_icd_adj)[enc_test_index]],
          (0,3): [to_dense(enc_pvd_adj)[enc_test_index]],
         }

y_train = y_label[enc_train_index]
y_val = y_label[enc_val_index]
y_test = y_label[enc_test_index]

'''
#icd
icd_val_idx  = np.where( np.isin(enc_icd.index, np.array(enc_val)))[0]
icd_test_idx =np.where(np.isin(enc_icd.index, np.array(enc_test)))[0] 
icd_train_idx = np.array([i for i in range(len(enc_icd.index)) if i not in icd_val_idx and i not in icd_test_idx ])
icd_train_val_idx = np.hstack((icd_train_idx,icd_val_idx))

train_mask_enc_icd=torch.zeros(enc_icd_adj.shape)
train_mask_enc_icd[icd_train_idx]=1
val_mask_enc_icd=torch.zeros(enc_icd_adj.shape)
val_mask_enc_icd[icd_val_idx]=1
test_mask_enc_icd=torch.zeros(enc_icd_adj.shape)
test_mask_enc_icd[icd_test_idx]=1
'''

tasks=((0,2),)

'''
train_adj_masks = {   (0, 2): [to_dense(train_mask_enc_icd)[enc_train_index].cuda()]}
val_adj_masks = {   (0, 2): [to_dense(train_mask_enc_icd)[enc_val_index].cuda()]}
test_adj_masks = {   (0, 2): [to_dense(train_mask_enc_icd)[enc_test_index].cuda()]}
'''

print("Preprocessing done in --- %s seconds ---" % (time.time() - start_time))
