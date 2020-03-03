#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import time
import pandas as pd
import numpy as np
import scipy as sp

from scipy import signal
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[2]:


def read_EMG(file_path, num): 
    
    coladd =  ['Time',
            'EMG_1', 
            'EMG_2', 
            'EMG_3', 
            'EMG_4', 
            'EMG_5', 
            'EMG_6', 
            'EMG_7', 
            'EMG_8']
    
    df = pd.read_csv(
                    file_path,
                    sep=r'\,|\t', 
                    engine='python',
                    header=None, 
                    names=coladd,
                    error_bad_lines=False)
    #Adding user information.  
    df['User'] = num
    
    coladd.pop(0)
    df[coladd].astype('int64')
    # will show up as NAN. Removing missing vals.
    df.dropna(how='any')
    
     
    return df

def read_ground(file_path): 
    
    df = pd.read_csv(file_path,
                     sep=r'\,|\t', 
                     engine='python',
                     header=None,
                     names=['start', 'end'],
                     usecols=[0,1])
    
    #Cleaning up groundTruth data. 
    #Converting values to int for easier mapping. 
    df.apply(lambda x: round(x*100 / 30))
    df.astype('int64')
    
    #Deleting na if in dataframe 
    df.dropna(how='any')
    
    return df
    


# In[3]:


def load_data(num): 

    num = str(num)
    if len(num) < 2:
        num = ''.join(['0', num])
        
    dir_path = 'C:/Users/crist/OneDrive/Desktop/Data Mining Assign 1/Data_Mining_Assign1Data/'
    
    #This code chunk will return groundTruth data for fork. 
    fl = [dir_path, 'groundTruth/user', num, '/fork/']
    fl_pth = ''.join(fl)
    for file in os.listdir(fl_pth):
        fl.append(file)
        file_path = ''.join(fl)
        ground_fork = read_ground(file_path)
    
    #This code chunk will return groundTruth data for spoon.
    fl = [dir_path, 'groundTruth/user', num, '/spoon/']
    fl_pth = ''.join(fl)
    for file in os.listdir(fl_pth):
        fl.append(file)
        file_path = ''.join(fl)
        ground_spoon = read_ground(file_path)
    
    
    #This code chunk will return EMG MyoData for fork. 
    fl = [dir_path, 'MyoData/user', num, '/fork/']
    fl_pth = ''.join(fl)
    for file in os.listdir(fl_pth):
        if file.endswith('_EMG.txt'): 
            fl.append(file)
            file_path = ''.join(fl)
            EMG_fork = read_EMG(file_path, num)
    
    #This code chunk will return EMG MyoData for spoon. 
    fl = [dir_path, 'MyoData/user', num, '/spoon/']
    fl_pth = ''.join(fl)
    for file in os.listdir(fl_pth):
        if file.endswith('_EMG.txt'): 
            fl.append(file)
            file_path = ''.join(fl)
            EMG_spoon = read_EMG(file_path, num)
    
    return ground_fork, ground_spoon, EMG_fork, EMG_spoon


# In[4]:


def combine_activity(ground_f, ground_s, emg_f, emg_s):
    
    #Map the groundTruth to all EMG sensor data for spoon and fork
    eat_f, eat_nof = map_data(ground_f, emg_f)
    eat_s, eat_nos = map_data(ground_s, emg_s)
    
    #Combine the mapped dataframes into two matrices.
    eat_df = pd.concat([eat_f, eat_s], axis=0)
    noeat_df = pd.concat([eat_nof, eat_nos], axis=0)
    
    #This will have all the activities in one matrix
    #will use this for the user independent analysis 
    comb = pd.concat([eat_df, noeat_df], axis=0)
        
    return comb, eat_df, noeat_df


# In[5]:


def map_data(ground_df, df): 

    #Setting all ACTIVITY to non-eating.
    df['ACTIVITY']= 0
    
    for index, row in ground_df.iterrows():
        start_frame = str(row[0])
        end_frame = str(row[1])
        try:         
            df.loc[start_frame:end_frame, 'ACTIVITY'] = 1
        except Exception as e:
            print(f'\n {e}.')
            
    eat = df[df.ACTIVITY == 1]
    noeat = df[df.ACTIVITY == 0]
    
    ex = len(eat.index)
    nx = len(noeat.index)
    if ex > nx: 
        size = int(nx)
    else: 
        size = int(ex)
            
    #Take equal number of eat/non-eat for equal class distrubution
    eat.sample(size)
    noeat.sample(size)
  
    return eat, noeat


# In[6]:


def get_min_max_std (df): 
    
    Sensors =  ['EMG_1', 
            'EMG_2', 
            'EMG_3', 
            'EMG_4', 
            'EMG_5', 
            'EMG_6', 
            'EMG_7', 
            'EMG_8']    
    
    smm = df[Sensors].describe()

    cat = [] 
    cat.append(smm.loc[['std']])
    cat[0].rename(columns={'EMG_1': 'EMG_1_STD', 
                           'EMG_2': 'EMG_2_STD',
                           'EMG_3': 'EMG_3_STD', 
                           'EMG_4': 'EMG_4_STD', 
                           'EMG_5': 'EMG_5_STD',
                           'EMG_6': 'EMG_6_STD', 
                           'EMG_7': 'EMG_7_STD', 
                           'EMG_8': 'EMG_8_STD'}, index={'std' : 0}, inplace=True)  
    cat.append(smm.loc[['min']])
    cat[1].rename(columns={'EMG_1': 'EMG_1_MIN', 
                           'EMG_2': 'EMG_2_MIN',
                           'EMG_3': 'EMG_3_MIN', 
                           'EMG_4': 'EMG_4_MIN', 
                           'EMG_5': 'EMG_5_MIN',
                           'EMG_6': 'EMG_6_MIN', 
                           'EMG_7': 'EMG_7_MIN', 
                           'EMG_8': 'EMG_8_MIN'}, index={'min' : 0}, inplace=True)
    cat.append(smm.loc[['max']])
    cat[2].rename(columns={'EMG_1': 'EMG_1_MAX', 
                           'EMG_2': 'EMG_2_MAX',
                           'EMG_3': 'EMG_3_MAX', 
                           'EMG_4': 'EMG_4_MAX', 
                           'EMG_5': 'EMG_5_MAX',
                           'EMG_6': 'EMG_6_MAX', 
                           'EMG_7': 'EMG_7_MAX', 
                           'EMG_8': 'EMG_8_MAX'}, index={'max' : 0}, inplace=True)
    #df with 24 columns of all sensors std, min, max
    smm = pd.concat(cat, axis=1)
    return smm

def get_rms(df): 
       
    Sensors =  ['EMG_1', 
                'EMG_2', 
                'EMG_3', 
                'EMG_4', 
                'EMG_5', 
                'EMG_6', 
                'EMG_7', 
                'EMG_8']
    
    df1 = df[Sensors].copy()
    rms_df = pd.DataFrame()
    
    for column in df1[Sensors]:
        df1[column] = df1[column].apply(lambda x: round(x**2))
        rms = df1[column].to_numpy()
        rms = np.mean(rms)
        rms = np.sqrt(rms)
        rms_df[f'{column}_RMS'] = [rms]
                
    return rms_df

def get_arv(df): 

    Sensors =  ['EMG_1', 
            'EMG_2', 
            'EMG_3', 
            'EMG_4', 
            'EMG_5', 
            'EMG_6', 
            'EMG_7', 
            'EMG_8']

    df1 = df.copy()
    df2 = pd.DataFrame()

    high = 6/(1000/2)
    low = 450/(1000/2)
    
    for column in df1[Sensors]: 
        emg = df1[column].to_numpy()
        emg_mean = emg - np.mean(emg) 
       
        #filtering for unwanted noise 
        b, a = sp.signal.butter(4, [high,low], btype='bandpass')
        #changed to lfilter because of padlen error 
        emg_filtered = sp.signal.lfilter(b, a, emg_mean)
        emg_rectified = abs(emg_filtered) 
        emg_rectified = np.mean(emg_rectified)        
        df2[f'{column}_ARV'] = [emg_rectified]

    return df2


# In[7]:


def slice_it(slice_df, size):
    return (slice_df[pos:pos + size] for pos in range(0, len(slice_df), size))

def prep_feat(df):

    l = []
    feat_df = pd.DataFrame() 
     
    for df_slice in slice_it(df, 100):
        df1 = get_min_max_std(df_slice)
        df2 = get_rms(df_slice)
        df1 = df1.join(df2, how='outer')
        df2 = get_arv(df_slice)
        df1 = df1.join(df2, how='outer')
        l.append(df1)
        
    feat_df = pd.concat(l)

    return feat_df


# In[8]:


def run_pca(feat_df): 
    
    activity = feat_df['ACTIVITY'].values
    feat_df.drop(['ACTIVITY'], axis=1, inplace=True)

    #selected two components most variance caputured within 2 
    pcamod = PCA(n_components=2)
    pca = pcamod.fit_transform(feat_df)
    
    feat_df = pd.DataFrame(data=pca, columns=['PC_1', 'PC_2'])    
    feat_df['ACTIVITY'] = activity
    
    return feat_df


# In[23]:


def tree_model(X_train, X_test, y_train, y_test): 

    #call in clf class and train model 
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    
    y_pred = tree_clf.predict(X_test)
    
    precision = round(precision_score(y_test, y_pred,average='weighted', labels=np.unique(y_pred)), 2)
    p = f'Precision: {precision}'
    
    recall = round(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 2)
    r = f'Recall: {recall}'
    
    F1 = round(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 2)
    f = f'F1: {F1}'
    
    
    return ('Decision Tree Metrics', p, r, f)


# In[24]:


def svm_model(X_train, X_test, y_train, y_test): 
    
    #trying LinearSVC apparently better for large datasets 
    svm_clf = LinearSVC(dual=False)
    svm_clf.fit(X_train, y_train)
    
    y_pred = svm_clf.predict(X_test)
    
    precision = round(precision_score(y_test, y_pred,average='weighted', labels=np.unique(y_pred)), 2)
    p = f'Precision: {precision}'
    
    recall = round(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 2)
    r = f'Recall: {recall}'
    
    F1 = round(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 2)
    f = f'F1: {F1}'
    
    
    return ('SVM Metrics', p, r, f)


# In[25]:


def nn_model(X_train, X_test, y_train, y_test):


    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),activation='logistic', max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    
    y_pred = mlp.predict(X_test)

    precision = round(precision_score(y_test, y_pred,average='weighted', labels=np.unique(y_pred)), 2)
    p = f'Precision: {precision}'
    
    recall = round(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 2)
    r = f'Recall: {recall}'
    
    F1 = round(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 2)
    f = f'F1: {F1}'
        
    return ('Neutral Network Metrics', p, r, f)


# In[12]:


def get_pca_df(df, activity):
    
    df1 = df[df.ACTIVITY == activity].copy()
    df1 = prep_feat(df1)
    df1['ACTIVITY'] = activity
    
    return run_pca(df1)


# In[13]:


def user_dep(test_list, train_list):
     
    train = pd.concat(train_list) 
    test = pd.concat(test_list)
    eat = 1
    noeat = 0 
    
    l = []
    l.append(get_pca_df(train, eat))
    l.append(get_pca_df(train, noeat))
    train_pca = pd.concat(l)

    lt = []
    lt.append(get_pca_df(test, eat))
    lt.append(get_pca_df(test, noeat))
    test_pca = pd.concat(lt)
    
    X_train = train_pca.drop('ACTIVITY', axis=1)
    y_train = train_pca['ACTIVITY']

    X_test = test_pca.drop('ACTIVITY', axis=1)
    y_test = test_pca['ACTIVITY']

    #run tree Model on all users
    t = tree_model(X_train, X_test, y_train, y_test)
   
    #run SVM model on all users 
    s = svm_model(X_train, X_test, y_train, y_test)
    
    #NN model on all users
    n = nn_model(X_train, X_test, y_train, y_test)

    return ('User Dependent Analysis', s, t, n)


# In[14]:


def user_indep(all_list): 
    
    df = pd.concat(all_list)
    eat = 1
    noeat = 0 
    
    l = []
    l.append(get_pca_df(df, eat))
    l.append(get_pca_df(df, noeat))
    all_pca = pd.concat(l)

    #Will need the following in all classifiers train/split all_pca dataframe
    X = all_pca.drop('ACTIVITY', axis=1)
    y = all_pca['ACTIVITY']
    
    #Split dataset into 60/40.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)
    
    #run tree Model on all users.
    t = tree_model(X_train, X_test, y_train, y_test)
   
    #run SVM model on all users.
    s = svm_model(X_train, X_test, y_train, y_test)
    
    #NN model on all users.
    n = nn_model(X_train, X_test, y_train, y_test)

    return ('User Independent Analysis', s, t, n)


# In[15]:


def split_train_test(e, n, split=.60): 
    
    #Calculating where I will be splitting the dataframes. 
    split_row = int(len(e.index) * split )
    
    #Making a train dataframe last .60 of combined data
    train = pd.concat([e.iloc[:split_row, :].copy(), n.iloc[:split_row, :].copy()], axis=0)
    
    #Making a test dataframe from the first .40 of the combined data. 
    test = pd.concat([e.iloc[split_row:, :].copy(), n.iloc[split_row:, :].copy()], axis=0)
    
    return train, test


# In[16]:


def display_report_accuracy(tup): 
    
    print(f'\n{tup[0]} Report\n')
    print(f'{tup[1][0]}\n{tup[1][1]}\t\t{tup[1][2]}\t\t{tup[1][3]}\n')
    print(f'{tup[2][0]}\n{tup[2][1]}\t\t{tup[2][2]}\t\t{tup[2][3]}\n')
    print(f'{tup[3][0]}\n{tup[3][1]}\t\t{tup[3][2]}\t\t{tup[3][3]}\n')
    


# In[17]:


def save_report_accuracy(a, b, filename='UserDep_UserIndep_Report.txt'): 
    
    ts = time.time()
    with open(filename, 'a+') as out: 
        out.write(f'\n{ts}\n')
        out.write(f'\n{a[0]} Report\n')
        out.write(f'{a[1][0]}\n{a[1][1]}\t\t{a[1][2]}\t\t{a[1][3]}\n')
        out.write(f'{a[2][0]}\n{a[2][1]}\t\t{a[2][2]}\t\t{a[2][3]}\n')
        out.write(f'{a[3][0]}\n{a[3][1]}\t\t{a[3][2]}\t\t{a[3][3]}\n')
        
        out.write(f'\n{b[0]} Report\n')
        out.write(f'{b[1][0]}\n{b[1][1]}\t\t{b[1][2]}\t\t{b[1][3]}\n')
        out.write(f'{b[2][0]}\n{b[2][1]}\t\t{b[2][2]}\t\t{b[2][3]}\n')
        out.write(f'{b[3][0]}\n{b[3][1]}\t\t{b[3][2]}\t\t{b[3][3]}\n')    
        


# In[26]:


#########################
# ASU CSE572 1216168421 #
# Maldonado, Cristina   #
# Assignment 2          #
#########################    

def main(): 
    
    all_list = []
    test_list = [] 
    train_list = [] 

    print('Loading data...')
    for i in range(9, 42):
        try:
            ground_f, ground_s, emg_f, emg_s = load_data(i) 
            comb, eat_df, noeat_df = combine_activity(ground_f, ground_s, emg_f, emg_s)
            train, test = split_train_test(eat_df, noeat_df)
            train_list.append(train)
            test_list.append(test)
            all_list.append(comb)

        except Exception as e:
            print(f'{e}\n')
            continue

    if test_list and train_list: 
        print('Running User dependent analysis...\n')
        ud_tup = user_dep(test_list, train_list)

    if all_list: 
        print('Running User independent analysis...\n')
        ui_tup = user_indep(all_list)
        
    print('Printing reports...\n')
    if ud_tup: 
        display_report_accuracy(ud_tup)

    if ui_tup:
        display_report_accuracy(ui_tup)

    print('Writing reports to file...\n')
    save_report_accuracy(ui_tup, ud_tup)
    

if __name__ == "__main__":
    main() 


# In[ ]:




