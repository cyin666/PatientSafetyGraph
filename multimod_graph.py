from utilities import *
from fastnode2vec import Graph, Node2Vec

import warnings
warnings.filterwarnings('ignore')

def multimod_graph_edglist_provider(df_master_preprocessed,\
                           df_provider_preprocessed):
    ###hospital_encounter + provider
   df_master_provider = pd.merge(df_master_preprocessed[['ENCRYPTED_HOSP_ENCOUNTER',\
                                                         'CASE_NUMBER']],\
                                 df_provider_preprocessed,\
                                 on='CASE_NUMBER',\
                                 how='left')
  # print(df_master_preprocessed.shape)
  # print(len(set(df_master_provider['ENCRYPTED_HOSP_ENCOUNTER'])))

   df_master_provider['ENCRYPTED_PROVIDER_NAME'].\
       replace(np.nan,'MISSING OR INVALID DATA FORMATION',inplace=True)
   df_master_provider['PROVIDER_TYPE'].\
       replace(np.nan,'Missing Role',inplace=True)
   df_master_provider = df_master_provider.\
                            loc[df_master_provider['ENCRYPTED_PROVIDER_NAME'] != \
                                                    "MISSING OR INVALID DATA FORMATION"].\
                            reset_index(drop=True)
  # print(df_master_preprocessed.shape)

   provider_edgelist = df_master_provider.\
                        groupby(['ENCRYPTED_HOSP_ENCOUNTER',\
                                 'ENCRYPTED_PROVIDER_NAME',\
                                 'PROVIDER_TYPE']).\
                        size().reset_index()
  # print(df_master_preprocessed.shape)
   provider_edgelist.columns = ['ENCRYPTED_HOSP_ENCOUNTER',\
                                 'ENCRYPTED_PROVIDER_NAME',\
                                 'PROVIDER_TYPE',\
                                 'weight']
  # print(len(set(df_master_provider['ENCRYPTED_HOSP_ENCOUNTER'])))
  # print(provider_edgelist.shape)
    
   provider_edgelist = pd.get_dummies(provider_edgelist,\
                                      columns=['PROVIDER_TYPE'],\
                                      drop_first=False)
 #  print(provider_edgelist.shape)
   provider_edgelist=provider_edgelist.\
                        groupby(['ENCRYPTED_HOSP_ENCOUNTER',\
                                 'ENCRYPTED_PROVIDER_NAME']).\
                        sum().reset_index()
  # print(provider_edgelist.shape)
  # print(provider_edgelist.columns)
        
   provider_edgelist['ENCRYPTED_PROVIDER_NAME'] = provider_edgelist['ENCRYPTED_PROVIDER_NAME'].\
                                                    replace('placeholder',np.nan)
   node_provider = list(set(provider_edgelist['ENCRYPTED_PROVIDER_NAME']))
  # print(len(node_provider))
  #print(len(set(provider_edgelist.ENCRYPTED_HOSP_ENCOUNTER)))
   
   node_feats_provider = provider_edgelist.\
                            loc[~provider_edgelist['ENCRYPTED_PROVIDER_NAME'].isna()].\
                            groupby('ENCRYPTED_PROVIDER_NAME').\
                            sum().drop(columns='weight').reset_index()

    
   return provider_edgelist,node_provider,node_feats_provider




##
def multimod_graph_edglist_icubed(df_master_preprocessed,\
                                  df_icu,\
                                  hospenc_index):
    df_master_preprocessed['Surgery_End_Datetime'] = pd.to_datetime(df_master_preprocessed['Surgery_End_Datetime'])
    
    df_icu_preOR = pd.DataFrame({'ENCRYPTED_HOSP_ENCOUNTER':[], 
                             'FROM_HOSP_ORG_LVL4_DESC':[], 
                             'FROM_BED_TYPE':[],
                             'ON_HOSP_ORG_LVL4_DESC':[],
                             'ON_BED_TYPE':[], 
                             'TO_HOSP_ORG_LVL4_DESC':[],
                             'TO_BED_TYPE':[], 
                             'ENTER_DT':[], 
                             'EXIT_DT':[], 
                             'TIME_IN_HOURS':[],
                             'TIME_IN_MINUTES':[]})
    
    for hosp_encounter_id in tqdm(hospenc_index):
        #print(hosp_encounter_id)
        df_preOR_icu_tmp = df_master_preprocessed.loc[df_master_preprocessed['ENCRYPTED_HOSP_ENCOUNTER'] == hosp_encounter_id].reset_index(drop=True)
    
        identify_time = df_preOR_icu_tmp['Surgery_End_Datetime'].values[0]
       # print(identify_time)
        subset_encounter_icu = df_icu[df_icu['ENCRYPTED_HOSP_ENCOUNTER'] == hosp_encounter_id].drop_duplicates()
        subset_encounter_icu['ENTER_DT'] = pd.to_datetime(subset_encounter_icu['ENTER_DT'])
        subset_encounter_icu_sorted = subset_encounter_icu.sort_values(by=['ENTER_DT'])

        subset_encounter_icu_clean = subset_encounter_icu_sorted[subset_encounter_icu_sorted['ENTER_DT'] < identify_time].\
                                        reset_index(drop=True)
        
        df_icu_preOR = pd.concat([df_icu_preOR,subset_encounter_icu_clean])

        df_icu_preOR = df_icu_preOR.reset_index(drop=True)
        
    df_icu_preOR = pd.merge(df_master_preprocessed[['ENCRYPTED_HOSP_ENCOUNTER']],\
                            df_icu_preOR,\
                            how='left',\
                            on='ENCRYPTED_HOSP_ENCOUNTER')
    df_icu_preOR['ON_HOSP_ORG_LVL4_DESC'].replace(np.nan,'placeholder',inplace=True)
    df_icu_preOR['ON_BED_TYPE'].replace(np.nan,'placeholder',inplace=True)
    

    icu_preOR_edgelist = df_icu_preOR.groupby(['ENCRYPTED_HOSP_ENCOUNTER',\
                                               'ON_HOSP_ORG_LVL4_DESC',\
                                               'ON_BED_TYPE']).\
                                      agg({'TIME_IN_MINUTES':['size','sum']}).\
                                      reset_index()

    icu_preOR_edgelist.columns = ['ENCRYPTED_HOSP_ENCOUNTER',\
                                  'bed_location_id',\
                                  'ON_BED_TYPE',\
                                  'weight',\
                                  'TIME_IN_MINUTES']
   
    
   # print(icu_preOR_edgelist['ON_BED_TYPE'].isna().sum())
    
    icu_preOR_edgelist = pd.get_dummies(icu_preOR_edgelist,columns=['ON_BED_TYPE'],drop_first=False)
  
    icu_preOR_edgelist['los_bed_acute'] = icu_preOR_edgelist['TIME_IN_MINUTES'] * icu_preOR_edgelist['ON_BED_TYPE_ACUTE']/60
    icu_preOR_edgelist['los_bed_intensive'] = icu_preOR_edgelist['TIME_IN_MINUTES'] * icu_preOR_edgelist['ON_BED_TYPE_INTENSIVE']/60
    icu_preOR_edgelist['los_bed_intermediate'] = icu_preOR_edgelist['TIME_IN_MINUTES'] * icu_preOR_edgelist['ON_BED_TYPE_INTERMEDIATE']/60
    
    icu_preOR_edgelist = icu_preOR_edgelist[['ENCRYPTED_HOSP_ENCOUNTER',\
                                            'bed_location_id',\
                                            'weight',\
                                            'los_bed_acute',\
                                            'los_bed_intensive',\
                                            'los_bed_intermediate']] 

    icu_preOR_edgelist=icu_preOR_edgelist.groupby(['ENCRYPTED_HOSP_ENCOUNTER','bed_location_id']).sum().reset_index()
    
   #print(icu_preOR_edgelist['bed_location_id'].isna().sum())
    
    icu_preOR_edgelist['bed_location_id'] = icu_preOR_edgelist['bed_location_id'].replace('placeholder',np.nan)
    node_icupreOR_bed = list(set(icu_preOR_edgelist['bed_location_id']))  
    
    node_feat_icu = icu_preOR_edgelist.\
                        loc[~icu_preOR_edgelist['bed_location_id'].isna()].\
                        groupby('bed_location_id').\
                        sum().\
                        drop(columns='weight').\
                        reset_index()
    
    return icu_preOR_edgelist,node_icupreOR_bed, node_feat_icu



##
def multimod_graph_edglist_icd(df_master_preprocessed,\
                                  df_comp_poa,\
                                  hospenc_index):
    #COMP FEATURES, node attribute: POA=present on admission (YES,NO,EXEMPT/?)
    hospenc_id = []
    top_level_icd_features_POAYes = []
    top_level_icd_features_POANo = []
    top_level_icd_features_POAExemptorUnknown = []
    for hosp_encounter_id in tqdm(hospenc_index):
      subset_encounter_comp = df_comp_poa[df_comp_poa['ENCRYPTED_HOSP_ENCOUNTER'] == hosp_encounter_id]
      icd_feature_POAYes = []
      icd_feature_POANo = []
      icd_feature_POAExemptorUnknown = []

      if len(subset_encounter_comp)!=0:
          icd_code_list = subset_encounter_comp.values.tolist()[0][1:] 
          for idx in range(1,len(icd_code_list),2):
            single_code = icd_code_list[idx-1]
            if single_code != '?':
                if icd_code_list[idx] in ['YES']:
                    icd_feature_POAYes.append(single_code[:3])
                if icd_code_list[idx] in ['NO']:
                    icd_feature_POANo.append(single_code[:3])
                if icd_code_list[idx] in ['EXEMPT','?']:
                    icd_feature_POAExemptorUnknown.append(single_code[:3])

      if(len(icd_feature_POAYes)==0):
        icd_feature_POAYes = ["placeholder"]
      if(len(icd_feature_POANo)==0):
        icd_feature_POANo = ["placeholder"]
      if(len(icd_feature_POAExemptorUnknown)==0):
        icd_feature_POAExemptorUnknown = ["placeholder"] #we still want the encounter in the dajacency matrix
        
      hospenc_id.append(hosp_encounter_id)
      top_level_icd_features_POAYes.append(icd_feature_POAYes)
      top_level_icd_features_POANo.append(icd_feature_POANo)
      top_level_icd_features_POAExemptorUnknown.append(icd_feature_POAExemptorUnknown)
        
    icd_edgelist_POAYes = pd.concat([concat_id_icd(hosp_encounter_id=x,icd_list=y,poa_status='YES') \
                                       for x, y in zip(hospenc_id,top_level_icd_features_POAYes)])
    icd_edgelist_POANo = pd.concat([concat_id_icd(hosp_encounter_id=x,icd_list=y,poa_status='NO') \
                                    for x, y in zip(hospenc_id,top_level_icd_features_POANo)])
    icd_edgelist_POAExemptorUnknown = pd.concat([concat_id_icd(hosp_encounter_id=x,icd_list=y,poa_status='POAExemptorUnknown') \
                                                 for x, y in zip(hospenc_id,top_level_icd_features_POAExemptorUnknown)])
    
    icd_edgelist = pd.concat([icd_edgelist_POAYes,icd_edgelist_POANo,icd_edgelist_POAExemptorUnknown])
    icd_edgelist = icd_edgelist.groupby(['ENCRYPTED_HOSP_ENCOUNTER','icd10_transformed','POA']).size().reset_index()
    icd_edgelist.columns = ['ENCRYPTED_HOSP_ENCOUNTER','icd10_transformed','node_attribute/POA','weight']

        #since icd is rolled up to high level, some of its original sub-icd may have POA="YES" while some may not, 
    #leading to "duplicate". So we one-hot and squeeze them to be node features
    icd_edgelist = pd.get_dummies(icd_edgelist,columns=['node_attribute/POA'],drop_first=False)
    icd_edgelist=icd_edgelist.groupby(['ENCRYPTED_HOSP_ENCOUNTER','icd10_transformed']).sum().reset_index()
    #remove "placeholder" to prevent unwanted node.
    icd_edgelist['icd10_transformed'] = icd_edgelist['icd10_transformed'].replace('placeholder',np.nan)
    node_icd_transformed = list(set(icd_edgelist['icd10_transformed']))
    
    node_feat_icd = icd_edgelist.\
                        loc[~icd_edgelist['icd10_transformed'].isna()].\
                        groupby('icd10_transformed').\
                        sum().\
                        drop(columns='weight').\
                        reset_index()
       
    return icd_edgelist,node_icd_transformed, node_feat_icd


##
def multimod_graph(df_master_preprocessed,\
                   df_provider_preprocessed,\
                   df_comp_poa,\
                   df_icu,\
                   hospenc_index):
    
    node_hospital_encounter_id = hospenc_index
    
    print("constructing edgelist of ICDs...")
    icd_edgelist,node_icd_transformed, node_feat_icd = multimod_graph_edglist_icd(df_master_preprocessed,\
                                                          df_comp_poa,\
                                                          hospenc_index)
    print("number of hosp_enc:", len(set(icd_edgelist['ENCRYPTED_HOSP_ENCOUNTER'])))
    
    print("constructing edgelist of preOR ICU beds...")
    icu_preOR_edgelist,node_icupreOR_bed, node_feat_icu = multimod_graph_edglist_icubed(df_master_preprocessed,\
                                                          df_icu,\
                                                          hospenc_index)
    print("number of hosp_enc:", len(set(icu_preOR_edgelist['ENCRYPTED_HOSP_ENCOUNTER'])))
    
    print("constructing edgelist of providers...")
    provider_edgelist,node_provider,node_feats_provider = multimod_graph_edglist_provider(df_master_preprocessed,\
                                                           df_provider_preprocessed)
    print("number of hosp_enc:", len(set(provider_edgelist['ENCRYPTED_HOSP_ENCOUNTER'])))
    
    print("constructing the multimodal graph...")
    icu_preOR_edgelist_final = icu_preOR_edgelist.iloc[:,0:3]
    icu_preOR_edgelist_final.columns = ['source','destination','weight']

    provider_edgelist_final = provider_edgelist.iloc[:,0:3]
    provider_edgelist_final.columns = ['source','destination','weight']

    icd_edgelist_final = icd_edgelist.iloc[:,0:3]
    icd_edgelist_final.columns = ['source','destination','weight']
    
    multimodal_graph_edgelist = pd.concat([icu_preOR_edgelist_final,\
                                           provider_edgelist_final,\
                                           icd_edgelist_final])
    print("number of hosp_enc:", len(set(multimodal_graph_edgelist['source'])))
    
    multimodal_graph = nx.from_pandas_edgelist(multimodal_graph_edgelist, \
                                               source='source', \
                                               target='destination', \
                                               edge_attr='weight', \
                                               create_using=nx.Graph)

    icd_edgelist_remove = icd_edgelist.\
                            loc[icd_edgelist['node_attribute/POA_YES']==0].\
                            iloc[:,0:2].\
                            drop_duplicates(ignore_index=True)
    icd_edgelist_remove.columns = ['source','destination']
    
    for index, row in icd_edgelist_remove.iterrows():
        multimodal_graph.remove_edge(row['source'],row['destination'])  
    
    
    nan_nodes = []
    for node in multimodal_graph.nodes():
        if isinstance(node, str): 
            continue
        if math.isnan(node):
            nan_nodes.append(node)
    multimodal_graph.remove_nodes_from(nan_nodes)
    
    node_icupreOR_bed = np.array(list(node_icupreOR_bed))
    node_icupreOR_bed = node_icupreOR_bed[node_icupreOR_bed!="nan"]
    node_icupreOR_bed = node_icupreOR_bed[~pd.isna(node_icupreOR_bed)]
    
    node_provider = np.array(node_provider)
    node_provider = node_provider[node_provider!="nan"]
    node_provider = node_provider[~pd.isna(node_provider)]
    
    node_hospital_encounter_id = np.array(node_hospital_encounter_id)
    node_hospital_encounter_id = node_hospital_encounter_id[node_hospital_encounter_id!="nan"]
    node_hospital_encounter_id = node_hospital_encounter_id[~pd.isna(node_hospital_encounter_id)]
    
    node_icd_transformed = np.array(node_icd_transformed)
    node_icd_transformed = node_icd_transformed[node_icd_transformed!="nan"]
    node_icd_transformed = node_icd_transformed[~pd.isna(node_icd_transformed)]
    
    
    for i in node_icupreOR_bed:
        multimodal_graph.nodes[i]['node_type'] = 'preOR_icu_bed_location'

    for i in node_provider:
        multimodal_graph.nodes[i]['node_type'] = 'provider_name'

    for i in node_icd_transformed:
        multimodal_graph.nodes[i]['node_type'] = 'truncated_icd10'

    for i in node_hospital_encounter_id:
        multimodal_graph.nodes[i]['node_type'] = 'hospital_encounter'
    
    print("Done!")
    
    return multimodal_graph, node_feat_icd, node_feat_icu, node_feats_provider