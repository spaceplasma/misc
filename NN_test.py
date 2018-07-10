# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:41:12 2018

@author: FEPC-T134
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:03:36 2018

@author: FEPC-T134
"""

import pandas as pd
import numpy as np 
import time

#from keras.wrappers.scikit_learn import KerasRegressor

#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout#, Flatten, Lambda, Activation#, Input
#from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint#, TensorBoard, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

#from sklearn import preprocessing

#from sklearn.model_selection import cross_val_score, cross_validate

#from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, median_absolute_error

from sklearn.model_selection import GroupKFold

#from sklearn.decomposition import PCA, KernelPCA

import matplotlib.pyplot as plt
#from scipy import stats

seed = 1337
np.random.seed(seed)

def combine_rfs(rf_a, rf_b):
    # This combines / ensembles the random forests
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

def random_forest_regressor(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestRegressor(bootstrap=False, min_samples_leaf=6, n_estimators=200, min_samples_split=24, criterion='mse', n_jobs=-1)
    clf.fit(features, np.log1p(target))
    return clf

def extra_trees_regressor(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = ExtraTreesRegressor(n_estimators=200, criterion = 'mse', bootstrap = False, min_samples_leaf = 3, min_samples_split = 10, n_jobs=-1)
    clf.fit(features, np.log1p(target))
    return clf

def catRegion(df):
    df_cat = df.copy()
    regions = ['Unknown','Eastern_Prairies', 'USA', 'Northern_Alberta', 'Central_Prairies', 'Southern_Alberta', 'Retired_Growers']
    cat_regions = [0, 1, 2, 3, 4, 5, 6]
    cr = 0
    df_cat[:] = 0
    for x in regions:
        df_cat[df == x] = cat_regions[cr]
        cr += 1
        
    return df_cat
    
def binEncode(df):
    df_bin = df.copy()
    df_bin[:] = 0
    df_bin[df == True] = 1
    
    return df_bin

def adjNCombined(df):
    
    depth = df['testDepthMax_sub']
    surf = df['n_surface_current'] 
    sub = df['n_sub_current']
    
    adj_N_combined = 2*surf + ((depth-6)/3)*sub
    
    return adj_N_combined


# define base model
def getBaselineModel():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=78, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    #myoptim=Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# define wide model
def getWideModel():
    # create model
    model = Sequential()
    model.add(BatchNormalization(input_shape=(78,)))
    #model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    #myoptim=Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# define Deep model
def getDeepModel():
    # create model
    model = Sequential()
    model.add(BatchNormalization(input_shape=(78,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))    
    model.add(Dense(64, activation='relu'))    
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    #myoptim=Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# define Deeper-Wider model
def getDeeperModel():
    # create model
    model = Sequential()
    model.add(BatchNormalization(input_shape=(78,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(64, activation='relu'))    
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    #myoptim=Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

### ---------
# Processing
### ----------
    
def cropEncoding(df, verbose=0):
    if verbose > 0: print('--> Cropping Encoding Function')
    encoded = df.copy()
    
    # set up crop type encodes
    # Legume; Cerals; Canola; Grass; Other; Unknown
    
    legume = ['Alfalfa__Hay','Alfalfa__Seed','Peas__Field__Yellow','Peas__Field__Green','Bean__Black','Bean__Cranberry',
          'Bean__Faba','Bean__Great_Northern','Bean__Navy','Bean__Pea_Navy','Bean__Pink','Bean__Pinto',
          'Bean__Red_Kidney','Bean__White_Kidney','Chickpea','Lentil','Soybean','Bean__Yellow','Grass__Legume_20-40%']
    
    grains = ['Barley__Feed','Barley__Malt','Buckwheat','Wheat__Hard_Spring_Wheat', 'Wheat__Durum', 'Wheat__Hard_Winter',
          'Wheat__CPS', 'Wheat__Special_Purpose', 'Wheat__Soft_Spring_Wheat', 'Wheat__ESRS', 'Wheat__Canadian_Western_Red_Spring',
          'Wheat__Northern_Hard_Red', 'Wheat__Soft_Winter', 'Unknown__Cereal', 'Triticale__Grain', 'Rye__Fall',
          'Oats', 'Oats_(Hay)', 'Flax', 'Sorghum','Rye__Spring', 'Silage__Cereal',
          'Millet', 'Oats__Hulless']

    grass = ['Grass__Fescue__Meadow_(Seed)','Grass__Fescue__Tall_(Seed)',
          'Grass__Hay','Grass__Native','Grass__Orchardgrass','Grass__Ryegrass_(Hay)','Grass__Ryegrass_(Seed)',
          'Grass__Timothy_(Hay)','Grass__Timothy_(Seed)','Grass__Wheatgrass_(Hay)','Grass__Fescue__Red_(Hay)']
    
    corn = ['Corn__Grain', 'Corn__Silage']
    
    other = ['Borage','Canaryseed__Annual', 'Coriander', 'Caraway' ,'Fallow', 'Green_Feed', 'Hemp__Seed', 'Hemp__Fiber',
         'Mustard__Indian_(Brown)','Mustard__White/Yellow', 'Other', 'Peola', 'Potato', 'Quinoa', 'Sugar_Beets',
         'Sunflower__Non-oilseed','Sunflower__Oilseed','Unknown__Forage','Cotton']
    
    encoded[df == 'Unknown'] = 0
    
    encoded[df == 'Canola'] = 1
    
    #Corn
    for x in corn:
        encoded[df == x] = 2
    
    #Legumes
    for x in legume:
        encoded[df == x] = 3
    
    #grains
    for x in grains:
        encoded[df == x] = 4

    #grass
    for x in grass:
        encoded[df == x] = 5
    
    #Other
    for x in other:
        encoded[df == x] = 6

    
    if verbose > 0: print('<-- Done Crop Encoding')
    
    return(encoded)
    
def preprocessingData(df):
    #print 'Pre-processing data for insertion'
    
    #known = adjNCombined(df[['testDepthMax_sub','n_surface_current','n_sub_current']])
    #known = df['n_combined_current']
    
    weather_cols = ['q_1_avgHumid', 'q_1_avgTemp', 'q_1_cloud', 'q_1_dewTemp',
           'q_1_dirNormIrrad', 'q_1_dwnRadiat', 'q_1_gdd', 'q_1_horRadiat',
           'q_1_maxPrecipEvent', 'q_1_precip', 'q_1_snow', 'q_1_wetTemp',
           'q_1_wind', 'q_2_avgHumid', 'q_2_avgTemp', 'q_2_cloud',
           'q_2_dewTemp', 'q_2_dirNormIrrad', 'q_2_dwnRadiat', 'q_2_gdd',
           'q_2_horRadiat', 'q_2_maxPrecipEvent', 'q_2_precip', 'q_2_snow',
           'q_2_wetTemp', 'q_2_wind', 'q_3_avgHumid', 'q_3_avgTemp',
           'q_3_cloud', 'q_3_dewTemp', 'q_3_dirNormIrrad', 'q_3_dwnRadiat',
           'q_3_gdd', 'q_3_horRadiat', 'q_3_maxPrecipEvent', 'q_3_precip',
           'q_3_snow', 'q_3_wetTemp', 'q_3_wind', 'q_4_avgHumid',
           'q_4_avgTemp', 'q_4_cloud', 'q_4_dewTemp', 'q_4_dirNormIrrad',
           'q_4_dwnRadiat', 'q_4_gdd', 'q_4_horRadiat', 'q_4_maxPrecipEvent',
           'q_4_precip', 'q_4_snow', 'q_4_wetTemp', 'q_4_wind']
    
    weather1 = ['q_1_avgHumid', 'q_1_avgTemp', 'q_1_cloud', 'q_1_dewTemp',
           'q_1_dirNormIrrad', 'q_1_dwnRadiat', 'q_1_gdd', 'q_1_horRadiat',
           'q_1_maxPrecipEvent', 'q_1_precip', 'q_1_snow', 'q_1_wetTemp',
           'q_1_wind']
    weather2 = ['q_2_avgHumid', 'q_2_avgTemp', 'q_2_cloud',
           'q_2_dewTemp', 'q_2_dirNormIrrad', 'q_2_dwnRadiat', 'q_2_gdd',
           'q_2_horRadiat', 'q_2_maxPrecipEvent', 'q_2_precip', 'q_2_snow',
           'q_2_wetTemp', 'q_2_wind']
    weather3 = ['q_3_avgHumid', 'q_3_avgTemp', 'q_3_cloud', 'q_3_dewTemp',
                'q_3_dirNormIrrad', 'q_3_dwnRadiat',
                'q_3_gdd', 'q_3_horRadiat', 'q_3_maxPrecipEvent', 'q_3_precip',
                'q_3_snow', 'q_3_wetTemp', 'q_3_wind']
    weather4 = ['q_4_avgHumid', 'q_4_avgTemp', 'q_4_cloud', 'q_4_dewTemp', 'q_4_dirNormIrrad',
           'q_4_dwnRadiat', 'q_4_gdd', 'q_4_horRadiat', 'q_4_maxPrecipEvent',
           'q_4_precip', 'q_4_snow', 'q_4_wetTemp', 'q_4_wind']
    
    predict_cols = ['n_surface_current', 'n_sub_current', 'n_combined_current',
           'p_surface_current', 'k_surface_current', 's_surface_current',
           's_sub_current', 's_combined_current']
    
    cols = ['region', 'farmID', 'managementZone_current',
           'managementZone_previous', 'season', 'zone', 'testDate_current',
           'testDate_previous', 'b_sub', 'ca_sub', 'cec_sub', 'cl_sub',
           'cu_sub', 'ec_sub', 'ec_ratio_sub', 'fe_sub',
           'isVirtual_sub_previous', 'k_sub', 'mg_sub', 'mn_sub',
           'n_sub_previous', 'na_sub', 'om_sub', 'p_sub', 'pH_sub',
           'pOther_sub', 's_sub_previous', 'testDepthMax_sub', 'zn_sub',
           'b_surface', 'ca_surface', 'cec_surface', 'cl_surface',
           'cu_surface', 'ec_surface', 'ec_ratio_surface', 'fe_surface',
           'isVirtual_surface_previous', 'k_surface_previous', 'mg_surface',
           'mn_surface', 'n_surface_previous', 'na_surface', 'om_surface',
           'p_surface_previous', 'pH_surface', 'pOther_surface',
           's_surface_previous', 'testDepthMax_surface', 'zn_surface',
           'latitude', 'longitude', 'continuousCropping', 'irrigated',
           'noTillage', 'previousStrawRemoved', 'probabilityOfPrecipitation',
           'strawRemoved', 'unitYieldTarget', 'cropName', 'cropName_previous',
           'checkArea', 'fieldArea', 'final_k', 'final_n', 'final_p',
           'final_s', 'previousYield', 'yieldTargetFinal', 'k_applied',
           'n_applied', 'p_applied', 's_applied', 'harvest_yield',
           'yield_units', 'previousSampleDay', 'currentSampleDay', 'timeSpan']
    
    stuff = ['region', 'farmID', 'managementZone_current',
           'managementZone_previous', 'season', 'zone', 'testDate_current',
           'testDate_previous', 'b_sub', 'ca_sub', 'cec_sub', 'cl_sub',
           'cu_sub', 'ec_sub', 'ec_ratio_sub', 'fe_sub',
           'isVirtual_sub_previous', 'k_sub', 'mg_sub', 'mn_sub',
           'n_sub_previous', 'na_sub', 'om_sub', 'p_sub', 'pH_sub',
           'pOther_sub', 's_sub_previous', 'testDepthMax_sub', 'zn_sub',
           'b_surface', 'ca_surface', 'cec_surface', 'cl_surface',
           'cu_surface', 'ec_surface', 'ec_ratio_surface', 'fe_surface',
           'isVirtual_surface_previous', 'k_surface_previous', 'mg_surface',
           'mn_surface', 'n_surface_previous', 'na_surface', 'om_surface',
           'p_surface_previous', 'pH_surface', 'pOther_surface',
           's_surface_previous', 'testDepthMax_surface', 'zn_surface',
           'latitude', 'longitude', 'continuousCropping', 'irrigated',
           'noTillage', 'previousStrawRemoved', 'probabilityOfPrecipitation',
           'strawRemoved', 'unitYieldTarget', 'cropName', 'cropName_previous',
           'checkArea', 'fieldArea', 'final_k', 'final_n', 'final_p',
           'final_s', 'previousYield', 'yieldTargetFinal', 'k_applied',
           'n_applied', 'p_applied', 's_applied', 'harvest_yield',
           'yield_units', 'previousSampleDay', 'currentSampleDay', 'timeSpan', 
           'q_1_avgHumid', 'q_1_avgTemp', 'q_1_cloud', 'q_1_dewTemp',
           'q_1_dirNormIrrad', 'q_1_dwnRadiat', 'q_1_gdd', 'q_1_horRadiat',
           'q_1_maxPrecipEvent', 'q_1_precip', 'q_1_snow', 'q_1_wetTemp',
           'q_1_wind', 'q_2_avgHumid', 'q_2_avgTemp', 'q_2_cloud',
           'q_2_dewTemp', 'q_2_dirNormIrrad', 'q_2_dwnRadiat', 'q_2_gdd',
           'q_2_horRadiat', 'q_2_maxPrecipEvent', 'q_2_precip', 'q_2_snow',
           'q_2_wetTemp', 'q_2_wind', 'q_3_avgHumid', 'q_3_avgTemp',
           'q_3_cloud', 'q_3_dewTemp', 'q_3_dirNormIrrad', 'q_3_dwnRadiat',
           'q_3_gdd', 'q_3_horRadiat', 'q_3_maxPrecipEvent', 'q_3_precip',
           'q_3_snow', 'q_3_wetTemp', 'q_3_wind', 'q_4_avgHumid',
           'q_4_avgTemp', 'q_4_cloud', 'q_4_dewTemp', 'q_4_dirNormIrrad',
           'q_4_dwnRadiat', 'q_4_gdd', 'q_4_horRadiat', 'q_4_maxPrecipEvent',
           'q_4_precip', 'q_4_snow', 'q_4_wetTemp', 'q_4_wind']
    
    changchi = ['region', 'farmID', 'managementZone_current',
           'managementZone_previous', 'season', 'zone', 'testDate_current',
           'testDate_previous', 'cl_sub',
           'ec_sub', 'ec_ratio_sub',
           'isVirtual_sub_previous', 
           'n_sub_previous', 'pH_sub',
           's_sub_previous', 'testDepthMax_sub',
           'b_surface', 'ca_surface', 'cec_surface', 'cl_surface',
           'cu_surface', 'ec_surface', 'ec_ratio_surface', 'fe_surface',
           'isVirtual_surface_previous', 'k_surface_previous', 'mg_surface',
           'mn_surface', 'n_surface_previous', 'na_surface', 'om_surface',
           'p_surface_previous', 'pH_surface',
           's_surface_previous', 'testDepthMax_surface', 'zn_surface',
           'latitude', 'longitude', 'continuousCropping', 'irrigated',
           'noTillage', 'previousStrawRemoved', 'probabilityOfPrecipitation',
           'strawRemoved', 'unitYieldTarget', 'cropName', 'cropName_previous',
           'checkArea', 'fieldArea', 'final_k', 'final_n', 'final_p',
           'final_s', 'k_applied',
           'n_applied', 'p_applied', 's_applied', 
           'previousSampleDay', 'currentSampleDay', 'timeSpan', 
           'q_1_avgHumid', 'q_1_avgTemp', 'q_1_cloud', 'q_1_dewTemp',
           'q_1_dirNormIrrad', 'q_1_dwnRadiat', 'q_1_gdd', 'q_1_horRadiat',
           'q_1_maxPrecipEvent', 'q_1_precip', 'q_1_snow', 'q_1_wetTemp',
           'q_1_wind', 'q_2_avgHumid', 'q_2_avgTemp', 'q_2_cloud',
           'q_2_dewTemp', 'q_2_dirNormIrrad', 'q_2_dwnRadiat', 'q_2_gdd',
           'q_2_horRadiat', 'q_2_maxPrecipEvent', 'q_2_precip', 'q_2_snow',
           'q_2_wetTemp', 'q_2_wind', 'q_3_avgHumid', 'q_3_avgTemp',
           'q_3_cloud', 'q_3_dewTemp', 'q_3_dirNormIrrad', 'q_3_dwnRadiat',
           'q_3_gdd', 'q_3_horRadiat', 'q_3_maxPrecipEvent', 'q_3_precip',
           'q_3_snow', 'q_3_wetTemp', 'q_3_wind', 'q_4_avgHumid',
           'q_4_avgTemp', 'q_4_cloud', 'q_4_dewTemp', 'q_4_dirNormIrrad',
           'q_4_dwnRadiat', 'q_4_gdd', 'q_4_horRadiat', 'q_4_maxPrecipEvent',
           'q_4_precip', 'q_4_snow', 'q_4_wetTemp', 'q_4_wind']
    
    df1 = df[cols]
    
    #drop all sub
    df1 = df1.drop(['b_sub', 'ca_sub', 'cec_sub', 'cu_sub', 'fe_sub', 'k_sub',
                   'mg_sub', 'mn_sub', 'na_sub', 'om_sub', 'p_sub', 'pOther_sub', 
                   'zn_sub', 'ec_ratio_sub', 'testDepthMax_sub', 'pH_sub', 
                   'cl_sub', 'ec_sub', 'pOther_surface', 'previousYield', 'harvest_yield',
                   'n_sub_previous', 's_sub_previous'],axis=1)
    
    #low variance
    df1 = df1.drop(['ec_ratio_surface', 'testDepthMax_surface', 'probabilityOfPrecipitation', 
                    'unitYieldTarget', 'checkArea', 'final_k', 'final_s'], axis=1)
 
    #df1 = df1.drop(['testDepthMax_surface', 'probabilityOfPrecipitation'], axis=1)

    
    #Zero - drop columns
    df1 = df1.drop(['final_n', 'final_p'],axis=1)
    
#    #Remove objects
    df1 = df1.drop(['testDate_current', 'testDate_previous'],axis=1)
    
    #Remove strings and unneed columns
    df1 = df1.drop(['farmID', 'managementZone_current', 'managementZone_previous', 'yield_units'], axis=1)
    
    # Binary / encoded columns
    df1 = df1.drop(['region', 'cropName', 'cropName_previous', 'strawRemoved', 'previousStrawRemoved',
                    'noTillage', 'irrigated', 'continuousCropping'], axis=1)
    
    # Multi-encoded categories
    #okay - use known values - unkowns are replaced by 0
    df_data_cat1 = catRegion(df['region'])
    
    # Binary Categories
    df_data_catbin1 = binEncode(df['strawRemoved'])
    df_data_catbin2 = binEncode(df['previousStrawRemoved'])
    df_data_catbin3 = binEncode(df['irrigated'])
    df_data_catbin4 = binEncode(df['continuousCropping'])
    
    cN = cropEncoding(df['cropName'])
    cN_prev = cropEncoding(df['cropName_previous'])
    
    # Let's say q3 weather is most important
    # By research, precip; temp
    #weather = df[['q_3_avgTemp','q_3_precip', 'q_2_avgTemp','q_2_precip']]
    weather = pd.concat([df[weather2],df[weather3]], axis=1)
    weather = weather.drop(['q_2_gdd', 'q_2_dirNormIrrad','q_3_gdd','q_3_snow','q_3_maxPrecipEvent','q_3_wetTemp',],axis=1)
    
    ### Concatenate the categorical values in
    cat = pd.concat([df_data_cat1,cN,cN_prev],axis=1)
    cat.columns = ['region_cat','cropName_cat','cropName_previous_cat']
    ohe1 = pd.get_dummies(cat)
    catbin = pd.concat([df_data_catbin1,df_data_catbin2,df_data_catbin3,df_data_catbin4], axis=1)
    catbin.columns=['strawRemoved_bin','previousStrawRemoved_bin','irrigated_bin','continuousCropping_bin']
    cat_encoded2 = df1['season']
    ohe2 = pd.get_dummies(cat_encoded2)
    cat_encoded3 = df1['zone']
    ohe3 = pd.get_dummies(cat_encoded3)
    df1 = df1.drop(['season','zone'],axis=1)
         
    df1 = pd.concat([df1,ohe1,ohe2,ohe3,catbin,weather], axis=1)
#    df1 = pd.concat([df1,cat,catbin], axis=1)
    
    ### Remove ROWS where fieldArea is 0 / nan
    df1['fieldArea'] = df1['fieldArea'].fillna(0)
    df1 = df1[df1.fieldArea != 0]
    
    ### Remove ROWS where weather is 0 / nan
    df1['q_3_avgTemp'] = df1['q_3_avgTemp'].fillna(0)
    df1 = df1[df1.q_3_avgTemp != 0]
    
#    ### Remove rows where zone # is wrong 
#    df1 = df1[df1.zone < 11]
#    
#    ### Remove rows where season is old
#    df1 = df1[df1.zone > 2013]
    
    ### Remove yieldTargetFinal
    df1 = df1.drop(['yieldTargetFinal'], axis=1)
    
    ### Engineer pH Nan / 0
    # pH of zero is bad choice as that is very acidic - set to 7 - neutral
    df1['pH_surface'] = df1['pH_surface'].fillna(7)
    
    df1['pH_surface'][df1['pH_surface'][df1['pH_surface']==0].index] = 7
    
    ### Remove ROWS where s_surface_previous == 60
    df1 = df1[df1.s_surface_previous != 60]
    
    ### Engineer Extreme Values where needed
    
#    xtreme_val = np.nanpercentile(np.array(df1['s_applied']), 99, axis=0)
    df1['s_applied'] = df1['s_applied'].fillna(np.nanmedian(np.array(df1['s_applied'])))
#    df1['s_applied'][df1['s_applied'] > xtreme_val] = xtreme_val
#    
#    xtreme_val = np.nanpercentile(np.array(df1['mn_surface']), 99, axis=0)
    df1['mn_surface'] = df1['mn_surface'].fillna(np.nanmedian(np.array(df1['mn_surface'])))
#    df1['mn_surface'][df1['mn_surface'] > xtreme_val] = xtreme_val
#    
#    xtreme_val = np.nanpercentile(np.array(df1['ec_surface']), 99, axis=0)
    df1['ec_surface'] = df1['ec_surface'].fillna(np.nanmedian(np.array(df1['ec_surface'])))
#    df1['ec_surface'][df1['ec_surface'] > xtreme_val] = xtreme_val
#    
#    xtreme_val = np.nanpercentile(np.array(df1['cl_surface']), 99, axis=0)
    df1['cl_surface'] = df1['cl_surface'].fillna(np.nanmedian(np.array(df1['cl_surface'])))
#    df1['cl_surface'][df1['cl_surface'] > xtreme_val] = xtreme_val
#    
#    xtreme_val = np.nanpercentile(np.array(df1['zn_surface']), 99, axis=0)
    df1['zn_surface'] = df1['zn_surface'].fillna(np.nanmedian(np.array(df1['zn_surface'])))
#    df1['zn_surface'][df1['zn_surface'] > xtreme_val] = xtreme_val
#    
#    xtreme_val = np.nanpercentile(np.array(df1['na_surface']), 99, axis=0)
    df1['na_surface'] = df1['na_surface'].fillna(np.nanmedian(np.array(df1['na_surface'])))
#    df1['na_surface'][df1['na_surface'] > xtreme_val] = xtreme_val
    
    ### Engineer 0 Values where needed
       
    #n_surface_previous null & 0
    df1['n_surface_previous'] = df1['n_surface_previous'].fillna(0)
    df1 = df1[df1.n_surface_previous != 0]
    
    df1 = df1.drop(['strawRemoved_bin',
           'previousStrawRemoved_bin', 'irrigated_bin'],axis=1)
    
    
    # set all null to 0 for minerals
    minerals = ['b_surface', 'ca_surface', 'cec_surface', 'cl_surface', 'cu_surface',
           'ec_surface', 'fe_surface', 'k_surface_previous', 'mg_surface',
           'mn_surface', 'na_surface', 'om_surface',
           'p_surface_previous', 'pH_surface', 's_surface_previous',
           'zn_surface']
    
    df1[minerals] = df1[minerals].fillna(0)
    
    # Now set all 0 to median values
    med_vals = df1[minerals].median()
    
    for x in minerals:
        df1[x][df1[x]==0] = med_vals[x]
    
    ### Drop lat / long for now -> from EDA
    #df1 = df1.drop(['latitude','longitude'],axis=1)
    
    npks_applied = ['n_applied','k_applied','p_applied','s_applied']
    
    ### change NPKS_applied values to 0 when null
    df1[npks_applied] = df1[npks_applied].fillna(0)
    
    df1 = df1.drop(['currentSampleDay','isVirtual_sub_previous','isVirtual_surface_previous'],axis=1)
    
    q = df1['previousSampleDay'][df1['previousSampleDay']<180]
    
    df1['previousSampleDay'].loc[q.index] = q+365
    
    #df1 = df1.drop(['previousSampleDay'],axis=1)
    
    # log1p feature transforms
    
    q = df1[['b_surface', 'cl_surface', 'cu_surface', 'ec_surface',
             'fe_surface', 'k_surface_previous', 'mg_surface', 'mn_surface',
             'n_surface_previous', 'na_surface', 'p_surface_previous', 's_surface_previous',
             'zn_surface', 'fieldArea', 'k_applied',
             'n_applied', 'p_applied', 's_applied', 'q_2_maxPrecipEvent', 'q_2_precip', 'q_2_snow',
             'q_2_wetTemp', 'q_2_wind', 'q_3_dirNormIrrad', 'q_3_dwnRadiat']]
    
    df1[q.columns.values] = np.log1p(q)
    
    known = df[['n_surface_current','p_surface_current','k_surface_current','s_surface_current']]
    known = known.loc[df1.index]
    
    MZID = df['managementZone_current']
    MZID = MZID[df1.index]
    
    df1 = pd.concat([MZID,known,df1],axis=1)
    
    ### from EDA
    df1 = df1.drop(['cec_surface'],axis=1)
    
    return df1

### ---
# Data
#
    
DATA_PATH = 'e:/Data/Demeter/'
df = pd.read_csv(DATA_PATH+str('TrainVST.csv'))
df_test = pd.read_csv(DATA_PATH+str('TestVST.csv'))

df1 = preprocessingData(df)
df1_test = preprocessingData(df_test)

df1 = df1.dropna(axis=0, how='any')

MZID_init = df1['managementZone_current']
npks_current = ['n_surface_current','p_surface_current','k_surface_current','s_surface_current']
#npks_current = ['n_surface_current']#,'p_surface_current','k_surface_current','s_surface_current']

known_all = df1[['n_surface_current','p_surface_current','k_surface_current','s_surface_current']]
X_df_init = df1.drop(['managementZone_current','n_surface_current',
                'p_surface_current','k_surface_current','s_surface_current'],axis=1)

df1_test = df1_test.dropna(axis=0, how='any')

known_test_all = df1_test[['n_surface_current','p_surface_current','k_surface_current','s_surface_current']]
X_df_test = df1_test.drop(['managementZone_current','n_surface_current',
                          'p_surface_current','k_surface_current','s_surface_current'],axis=1)
        
    
# Since test values don't change with fold, normalize them here only once
# using MaxMinScaler won't affect the OHE values on range 0,1
    
test_scale = MinMaxScaler()
X_df_test_sc = test_scale.fit_transform(X_df_test)



# Split Data
#start_time = time.time()
splits = 3
i = 1
batch_size = 16
epochs = 20
pred_ens = []
y_true = []
cv_medAE = []


for x in npks_current:
    
    print '\n',x

    known = (known_all[x])

    known = np.log1p(known)
    known_test = (known_test_all[x])
    
    known = known.replace(0,np.nan)
    
    q = known[~known.isnull()].index
    known = known.loc[q]
    MZID = MZID_init.loc[q]
    X_df = X_df_init.loc[q]
    
    old_medae = 99
    
    group_kfold = GroupKFold(n_splits=splits)
    #model = getBaselineModel()
    model = getDeeperModel()
    #model = getWideModel()
    #model = getDeepModel()
    
    if i == 1:
        print(model.summary())
    
    print('\nBaseline Model - 5-Fold - Run #'+str(i))
    for train_index, cv_index in group_kfold.split(X_df, known, MZID):
    
        X_train, X_cv = (np.array(X_df))[train_index], (np.array(X_df))[cv_index]
        y_train, y_cv = (np.array(known))[train_index], (np.array(known))[cv_index]

        # Need to normalize each fold separately
        fold_scale = MinMaxScaler()
        X_train_sc = fold_scale.fit_transform(X_train)
        
        cv_scale = MinMaxScaler()
        X_cv_sc = cv_scale.fit_transform(X_cv)

        earlyStopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('.mdl_wts'+str(i)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2, verbose=1, epsilon=1e-4, mode='min')


        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                  callbacks=[earlyStopping, mcp_save],validation_data=(X_cv,y_cv))
        
        model.load_weights(filepath = '.mdl_wts'+str(i)+'.hdf5')
             
        y_true = np.hstack((y_true,y_cv))
        pred_cv = model.predict(X_cv_sc) 
        pred_ens = np.hstack((pred_ens,pred_cv[:,0]))
        
#        medae = median_absolute_error(y_cv,pred_cv)
#        cv_medAE.append(medae)
#        
#
#        #print(medae,old_medae)    
#        # Find best model based on CV
#        if medae < old_medae:
#            old_medae = medae
#            best_model = model
            
            
    i = i+1
    
    print(x+' - cv')
    abs_err = abs(np.expm1(y_true) - np.expm1(pred_ens))
    print('\n50%: ',np.percentile(abs_err,50))
    print('80%: ',np.percentile(abs_err,80))
    print('95%: ',np.percentile(abs_err,95))

    
    