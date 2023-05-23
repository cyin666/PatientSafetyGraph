import argparse

import torch

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utility.preprocessing import sparse_to_tensor, to_dense
from utility.rank_metrics import *
from utility.iofile import *
from sklearn.metrics import label_ranking_average_precision_score, mean_squared_error

from model import MedGCN, MultiMatLoss 

from preparedata import * 

parser = argparse.ArgumentParser(description='Training and evaluating the MedGCN')
parser.add_argument('--lamb', type=float, default=1, help='regularization parameter for lab error')
parser.add_argument('--layers', type=int, default=1, help='GCN layers')
args = parser.parse_args()

torch.manual_seed(123)

epochs=100
# n_iter_no_change=50

adj_losstype = { (0, 1): [('BCE',0)],   (0, 2): [('MSE',args.lamb)],   (0, 3): [('BCE',1)] }

medgcn =  MedGCN(fea_num, ( 300,  )*args.layers, 1, tasks, dropout=0.0).cuda()
optimizer = optim.Adam(medgcn.parameters(), lr=1e-3)

# change loss function to cross entropy
lossfun = nn.BCELoss()

# lossfun = MultiMatLoss(pos_weight=None).cuda()
# scheduler = ReduceLROnPlateau(optimizer, 'min')

# training-validation-test
tr_loss, v_loss = [],[]
v_mapk, te_mapk = [],[]
v_lrap, te_lrap = [],[]
v_mse, te_mse = [],[]
best_val_loss = np.inf
best_val_lrap, best_val_mse = 0, np.inf
best_epoch = 0
for epoch in range(epochs):
    #training
    print("training............")
    medgcn.train()
    
    optimizer.zero_grad()
    y_train_pred, z = medgcn(train_fea_mats, train_adj_mats, train_adj_masks )
    #train_loss=lossfun(adj_recon, adj_mats, train_adj_masks, adj_losstype)
          
    train_loss = lossfun(y_train_pred[(0,2)][0], y_train.unsqueeze(1))
    train_loss.backward()
    optimizer.step()
    
    y_train_acc = (y_train_pred[(0,2)][0].reshape(-1).detach().cpu().numpy().round() == y_train.cpu().numpy()).sum() / y_train.cpu().numpy().shape[0]
    
    # validation-test
    print("validation............")
    medgcn.eval()
    with torch.no_grad():
        y_val_pred, z = medgcn(val_fea_mats, val_adj_mats, val_adj_masks )
        y_test_pred, z = medgcn(test_fea_mats, test_adj_mats, test_adj_masks )
                
        y_val_acc = (y_val_pred[(0,2)][0].reshape(-1).detach().cpu().numpy().round() == y_val.cpu().numpy()).sum() / y_val.cpu().numpy().shape[0]
        
        y_test_acc = (y_test_pred[(0,2)][0].reshape(-1).detach().cpu().numpy().round() == y_test.cpu().numpy()).sum() / y_test.cpu().numpy().shape[0]
        
        val_loss = lossfun(y_val_pred[(0,2)][0], y_val.unsqueeze(1))
        test_loss = lossfun(y_test_pred[(0,2)][0], y_test.unsqueeze(1))
    
    print('====> Epoch: {} train loss: {:.4f}, validation loss: {:.4f}, test loss: {:.4f}, training accuracy: {:.4f}, validation accuracy: {:.4f}, test accuracy: {:.4f}'  \
          .format(epoch, train_loss.item(), val_loss.item(), test_loss.item(),y_train_acc.item(), y_val_acc.item(), y_test_acc.item()))
    
    tr_loss.append(train_loss.item())
    v_loss.append(val_loss.item())
    
    
        
###### early stopping
#     if val_loss<=best_val_loss:
#         best_val_loss=val_loss
#         best_epoch=epoch
        
#     if test_mse<=best_val_mse:
#         best_val_mse=test_mse
#         save_obj(adj_recon[(0,2)][0].cpu().numpy(), 'est_lab2.pkl')
#         best_epoch=epoch
        
#     if test_lrap>=best_val_lrap:
#         best_val_lrap=test_lrap
#         save_obj(adj_recon[(0,3)][0].cpu().numpy(), 'rec_med2.pkl')
#         best_epoch=epoch

#     if epoch-best_epoch>n_iter_no_change:
#         break
    
res=pd.DataFrame({   'tr_loss': tr_loss, 'v_loss': v_loss, 
                  'v_mapk': v_mapk,'te_mapk': te_mapk, 'v_lrap': v_lrap, 'te_lrap': te_lrap,
                  'v_mse': v_mse, 'te_mse': te_mse
                 })
# res.to_csv('res_Med_lab%s.csv'%(args.lamb))
