#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:18:51 2023

@author: robertovincis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:39:33 2022

@author: robertovincis
"""

def extr_responsivness_wilcoxon_and_tuning(file):

    #from VincisLab_Python.EnsembleDecoding.decoding_tools import bin_all_spike_trains
    from Script.Script import bin_all_spike_trains
    import numpy as np
    import pickle
    import pandas as pd
    import scipy.stats as st
    from pathlib import Path
    from sklearn.preprocessing import normalize
    
    filePath = file

    # load the summary dataframe 
    fileName_main = 'Main_Temp.csv'
    Main_Temp = pd.read_csv(filePath / fileName_main) # load the main dataframe
    fileName = 'SU_Analysis/allN.pickle' 
    
    #Tuning_2 = []
    Tuning_3 = []
    for n in range(len(Main_Temp)):
        if Main_Temp.loc[n,'flag_Temp']== True:
            mouse = Main_Temp.loc[n,'mouse']
            if len(str(Main_Temp.loc[n,'date']))<6:
                Date='0' + str(Main_Temp.loc[n,'date'])
            else:
                Date=str(Main_Temp.loc[n,'date'])
            neuron_str=Main_Temp.loc[n,'name Wav']
            neuron = neuron_str.replace('waveforms','SU')
            neuron = neuron.replace('npy','csv')
            with open(filePath / mouse / Date / fileName,'rb') as file: 
                allN = pickle.load(file)
            df = allN[allN['n_ID']==neuron]
            min_index = 7
            Tuning_2_temp = []
            Tuning_3_temp = []
            for t in range(len(df['Taste'].unique())):
                df_T = df[df['Taste']==t]
                df_binned = bin_all_spike_trains(df_T, bin_width=100)
                df_binned.drop(df_binned.iloc[:, :min_index], inplace=True, axis=1)
                df_binned = df_binned.div(0.1)
                test = df_binned.iloc[:,5:20].mean()
                test_2 = df_binned.iloc[:,20:35].mean()
                #test_3=pd.concat([test,test_2])
                # w_res = st.wilcoxon(test,test_2)
                # if w_res.pvalue<0.05:
                #     Tuning_2_temp.append(1)
                # else:
                #     Tuning_2_temp.append(0)
                if test_2.mean()>test.mean():
                    Tuning_3_temp.append(max(test_2))                    
                else:
                    #Tuning_3_temp.append(min(test_2))
                    Tuning_3_temp.append(test_2[test_2>0.01].min())
            
            Tuning_3a_temp = normalize([Tuning_3_temp],"max")
            #Tuning_2.append(Tuning_2_temp)  
            Tuning_3.append(list(Tuning_3a_temp[0]))  

        else:
            Tuning_2_temp = [0, 0, 0]
            Tuning_3_temp = [0, 0, 0]
            #Tuning_2.append(Tuning_2_temp) 
            Tuning_3.append(Tuning_3_temp)  

    #Fin   = np.array(Tuning_2)
    Fin_2 = np.array(Tuning_3)
    
    return Fin_2


def prepare_for_decoding_zscore(Resp_sel_final):

    Resp_sel_final_reset = Resp_sel_final.reset_index()
    import pickle
    import pandas as pd
    #import scipy.stats as st
    from pathlib import Path
    from VincisLab_Python.EnsembleDecoding.decoding_tools import bin_all_spike_trains

    #from sklearn.preprocessing import normalize
    
    filePath = Path('/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Data_for_analysis') 
    fileName = 'SU_Analysis/allN.pickle' 
    
    for n in range(len(Resp_sel_final)):
        mouse = Resp_sel_final_reset.loc[n,'mouse']
        if len(str(Resp_sel_final_reset.loc[n,'date']))<6:
            Date='0' + str(Resp_sel_final_reset.loc[n,'date'])
        else:
            Date=str(Resp_sel_final_reset.loc[n,'date'])
        neuron_str=Resp_sel_final_reset.loc[n,'name Wav']
        neuron = neuron_str.replace('waveforms','SU')
        neuron = neuron.replace('npy','csv')
        with open(filePath / mouse / Date / fileName,'rb') as file: 
            allN = pickle.load(file)
        df = allN[allN['n_ID']==neuron]
        df['Neuron']=n
        if n==0:
            df_dec = df
        else:
            df_dec = pd.concat([df_dec,df],axis=0)
    
    df_dec_binned = bin_all_spike_trains(df_dec, 10) 
    for n in df_dec_binned['Neuron'].unique():
        df_temp_1 = df_dec_binned[df_dec_binned['Neuron']==n]
        df_temp_1_copy = df_temp_1.copy()
        df_temp_1_copy.drop(df_temp_1_copy.iloc[:,7:17],inplace=True, axis=1)
        shuffle_labels = pd.DataFrame()
        for s in df_temp_1['Taste'].unique():
            df_temp_2 = df_temp_1_copy[df_temp_1['Taste']==s]
            bas_M = df_temp_2.loc[:,-1000:0].mean().mean()
            bas_st = df_temp_2.loc[:,-1000:0].std().std()
            df_temp_2_copy = df_temp_2.copy()
            if bas_st == 0:
                df_temp_2_copy.iloc[:,7:] = (df_temp_2.iloc[:,7:]-bas_M)/1
            else:
                df_temp_2_copy.iloc[:,7:] = (df_temp_2.iloc[:,7:]-bas_M)/bas_st
            if s == 0:
                x = df_temp_2_copy
            else:
                x = pd.concat([x,df_temp_2_copy],axis=0)
        for sh in range(100):
            shuffle_labels.loc[:,sh] = x['Taste'].sample(frac=1).reset_index(drop=True)
            
        if n == 0:
            y = x
            y_shuffle = shuffle_labels
        else:
            y = pd.concat([y,x])
            y_shuffle = pd.concat([y_shuffle,shuffle_labels])
            
    return y,y_shuffle

def prepare_for_decoding(Resp_sel_final, bin_size = 50):

    Resp_sel_final_reset = Resp_sel_final.reset_index()
    import pickle
    import pandas as pd
    #import scipy.stats as st
    from pathlib import Path
    from Script import bin_all_spike_trains
    
    #from sklearn.preprocessing import normalize
    
    filePath = Path('/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Data_for_analysis') 
    fileName = 'SU_Analysis/allN.pickle' 
    
    for n in range(len(Resp_sel_final)): # to change back
        mouse = Resp_sel_final_reset.loc[n,'mouse']
        if len(str(Resp_sel_final_reset.loc[n,'date']))<6:
            Date='0' + str(Resp_sel_final_reset.loc[n,'date'])
        else:
            Date=str(Resp_sel_final_reset.loc[n,'date'])
        neuron_str=Resp_sel_final_reset.loc[n,'name Wav']
        neuron = neuron_str.replace('waveforms','SU')
        neuron = neuron.replace('npy','csv')
        with open(filePath / mouse / Date / fileName,'rb') as file: 
            allN = pickle.load(file)
        df = allN[allN['n_ID']==neuron]
        df['Neuron']=n
        if n==0:
            df_dec = df
        else:
            df_dec = pd.concat([df_dec,df],axis=0)
            
    df_dec_binned = bin_all_spike_trains(df_dec, bin_size)
    
    for n in df_dec_binned['Neuron'].unique():
        df_temp_1 = df_dec_binned[df_dec_binned['Neuron']==n]
        shuffle_labels = pd.DataFrame()
        for sh in range(100):
            shuffle_labels.loc[:,sh] = df_temp_1['Taste'].sample(frac=1).reset_index(drop=True)            
        if n == 0:
            y_shuffle = shuffle_labels
        else:
            y_shuffle = pd.concat([y_shuffle,shuffle_labels])
    y_Shuffle = y_shuffle.reset_index(drop=True)      
    return df_dec_binned,y_Shuffle

def prepare_for_decoding_water(Resp_sel_final, bin_size = 50):

    Resp_sel_final_reset = Resp_sel_final.reset_index()
    import pickle
    import pandas as pd
    #import scipy.stats as st
    from pathlib import Path
    from VincisLab_Python.EnsembleDecoding.decoding_tools import bin_all_spike_trains
    
    #from sklearn.preprocessing import normalize
    
    filePath = Path('/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Data_for_analysis') 
    fileName = 'SU_Analysis/allN.pickle' 
    
    for n in range(len(Resp_sel_final)): # to change back
        mouse = Resp_sel_final_reset.loc[n,'mouse']
        if len(str(Resp_sel_final_reset.loc[n,'date']))<6:
            Date='0' + str(Resp_sel_final_reset.loc[n,'date'])
        else:
            Date=str(Resp_sel_final_reset.loc[n,'date'])
        neuron_str=Resp_sel_final_reset.loc[n,'name Wav']
        neuron = neuron_str.replace('waveforms','SU')
        neuron = neuron.replace('npy','csv')
        with open(filePath / mouse / Date / fileName,'rb') as file: 
            allN = pickle.load(file)
            
        allN_F=allN[(allN['Taste']==2) | (allN['Taste']==3)]
        dict = {2:0,
                3:1}
        allN_F['Taste'].replace(dict, inplace=True)
        
        df = allN_F[allN_F['n_ID']==neuron]
        df['Neuron']=n
        if n==0:
            df_dec = df
        else:
            df_dec = pd.concat([df_dec,df],axis=0)
            
    df_dec_binned = bin_all_spike_trains(df_dec, bin_size)
    
    for n in df_dec_binned['Neuron'].unique():
        df_temp_1 = df_dec_binned[df_dec_binned['Neuron']==n]
        shuffle_labels = pd.DataFrame()
        for sh in range(100):
            shuffle_labels.loc[:,sh] = df_temp_1['Taste'].sample(frac=1).reset_index(drop=True)            
        if n == 0:
            y_shuffle = shuffle_labels
        else:
            y_shuffle = pd.concat([y_shuffle,shuffle_labels])
    y_Shuffle = y_shuffle.reset_index(drop=True)      
    return df_dec_binned,y_Shuffle

def prepare_for_decoding_saliva(Resp_sel_final, bin_size = 50):

    Resp_sel_final_reset = Resp_sel_final.reset_index()
    import pickle
    import pandas as pd
    #import scipy.stats as st
    from pathlib import Path
    from VincisLab_Python.EnsembleDecoding.decoding_tools import bin_all_spike_trains
    
    #from sklearn.preprocessing import normalize
    
    filePath = Path('/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Data_for_analysis') 
    fileName = 'SU_Analysis/allN.pickle' 
    
    for n in range(len(Resp_sel_final)): # to change back
        mouse = Resp_sel_final_reset.loc[n,'mouse']
        if len(str(Resp_sel_final_reset.loc[n,'date']))<6:
            Date='0' + str(Resp_sel_final_reset.loc[n,'date'])
        else:
            Date=str(Resp_sel_final_reset.loc[n,'date'])
        neuron_str=Resp_sel_final_reset.loc[n,'name Wav']
        neuron = neuron_str.replace('waveforms','SU')
        neuron = neuron.replace('npy','csv')
        with open(filePath / mouse / Date / fileName,'rb') as file: 
            allN = pickle.load(file)
            
        allN_F=allN[(allN['Taste']==0) | (allN['Taste']==1)]
        allN_F['Taste'].replace(dict, inplace=True)
        
        df = allN_F[allN_F['n_ID']==neuron]
        df['Neuron']=n
        if n==0:
            df_dec = df
        else:
            df_dec = pd.concat([df_dec,df],axis=0)
            
    df_dec_binned = bin_all_spike_trains(df_dec, bin_size)
    
    for n in df_dec_binned['Neuron'].unique():
        df_temp_1 = df_dec_binned[df_dec_binned['Neuron']==n]
        shuffle_labels = pd.DataFrame()
        for sh in range(100):
            shuffle_labels.loc[:,sh] = df_temp_1['Taste'].sample(frac=1).reset_index(drop=True)            
        if n == 0:
            y_shuffle = shuffle_labels
        else:
            y_shuffle = pd.concat([y_shuffle,shuffle_labels])
    y_Shuffle = y_shuffle.reset_index(drop=True)      
    return df_dec_binned,y_Shuffle

def sliding_decoding_Bouaichi_Neese_Vincis(y,y_shuffle):
    
    from Script import sliding_window_decoding
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    
    def createList(r1, r2):
        return np.arange(r1, r2+1, 1)
    
    mean_SVM_df_shuffle = pd.DataFrame()
    Y = y
    for i in tqdm(range(10)):
        temp_y = Y
        temp_y['Taste'] = y_shuffle.iloc[:,i]
        Temporary_df_all = pd.DataFrame()
        for n in temp_y['Neuron'].unique():
            Temporary_df = pd.DataFrame()
            a = temp_y[temp_y['Neuron']==n]
            one = a[a['Taste']==0]
            two = a[a['Taste']==1]
            three = a[a['Taste']==2]
            one['Trial'] = createList(0, len(one)-1)
            two['Trial'] = createList(0, len(two)-1)
            three['Trial'] = createList(0, len(three)-1)
            Temporary_df = pd.concat([one,two,three],axis=0).reset_index(drop=True)
            if n == 0:
                Temporary_df_all = Temporary_df
            else:
                Temporary_df_all = pd.concat([Temporary_df_all,Temporary_df],axis=0).reset_index(drop=True)
                
        mean_SVM_df, splits_SVM_df = sliding_window_decoding(Temporary_df_all, method = 'ensemble', ensemble_averaging=False,window_size=2, plot_confusion_matrix=False,class_labels=['Cold','Hot','Room'])
        mean_SVM_df_shuffle = pd.concat([mean_SVM_df_shuffle,mean_SVM_df.iloc[0,1:].to_frame().T],axis = 0,ignore_index=True) 
    mean_SVM_df_shuffle.to_csv('/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Data_for_analysis/shuffled_slidwin_decoding.csv')
    return mean_SVM_df_shuffle

def ensemble_decoding_Bouaichi_Neese_Vincis(y):
    
    from Script import ensemble_decoding
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    
    def createList(r1, r2):
        return np.arange(r1, r2+1, 1)
    
    # select only one second of post stimulus activity (here the spiking activity has been binned into 10ms bins already)
    #y_copy = y.copy()
    #min_index = y_copy.columns.get_loc('Trial') + 1
    #y_copy.drop(y_copy.iloc[:, min_index:207], inplace=True, axis=1)
    #y_copy.drop(y_copy.iloc[:, 108:207], inplace=True, axis=1)
    
    #y_copy = y.copy()
    #min_index = y_copy.columns.get_loc('Trial') + 1
    #y_copy.drop(y_copy.iloc[:, min_index:27], inplace=True, axis=1)
    #y_copy.drop(y_copy.iloc[:, 23:], inplace=True, axis=1)
    
    tot_list_neuron = list(y['Neuron'].unique())
    neur = []
    dec_values = []

    for n in tot_list_neuron:
    #for n in range(10):
        neur.append(n)
        temp_df = y.loc[y['Neuron'].isin(neur)]
        for N in temp_df['Neuron'].unique():
            Temporary_df = pd.DataFrame()
            a = temp_df[temp_df['Neuron']==N]
            one = a[a['Taste']==0]
            two = a[a['Taste']==1]
            three = a[a['Taste']==2]
            one['Trial'] = createList(0, len(one)-1)
            two['Trial'] = createList(0, len(two)-1)
            three['Trial'] = createList(0, len(three)-1)
            Temporary_df = pd.concat([one,two,three],axis=0).reset_index(drop=True)
            if N == 0:
                Temporary_df_all = Temporary_df
            else:
                Temporary_df_all = pd.concat([Temporary_df_all,Temporary_df],axis=0).reset_index(drop=True)
                
        test,test_2 = ensemble_decoding(Temporary_df_all, ensemble_averaging=False, plot_confusion_matrix=False, cm_plot_title=None, n_trial_pairings=50, test_size=1/3, num_splits=20, class_labels=['Cold','Hot','Room'])
        dec_values.append(test.iloc[0,1])
    return dec_values
        
def ensemble_decoding_pre_Bouaichi_Neese_Vincis(y):
    
    from VincisLab_Python.EnsembleDecoding.decoding_tools import ensemble_decoding
    import pandas as pd
    import numpy as np
    
    def createList(r1, r2):
        return np.arange(r1, r2+1, 1)
    
    # select only one second of post stimulus activity (here the spiking activity has been binned into 10ms bins already)
    # y_copy = y.copy()
    # min_index = y_copy.columns.get_loc('Trial') + 1
    # #y_copy.drop(y_copy.iloc[:, min_index:107], inplace=True, axis=1)
    # y_copy.drop(y_copy.iloc[:, 107:], inplace=True, axis=1)
    
    # y_copy = y.copy()
    # min_index = y_copy.columns.get_loc('Trial') + 1
    # #y_copy.drop(y_copy.iloc[:, min_index:107], inplace=True, axis=1)
    # y_copy.drop(y_copy.iloc[:, 26:], inplace=True, axis=1)
    #y_copy = y.copy()
    #min_index = y_copy.columns.get_loc('Trial') + 1
    #y_copy.drop(y_copy.iloc[:, min_index:7], inplace=True, axis=1)
    #y_copy.drop(y_copy.iloc[:, 27:], inplace=True, axis=1)

    tot_list_neuron = list(y['Neuron'].unique())
    neur = []
    dec_values_pre = []


    for n in tot_list_neuron:
#    for n in range(131):
        neur.append(n)
        temp_df = y.loc[y['Neuron'].isin(neur)]
        for N in temp_df['Neuron'].unique():
            Temporary_df = pd.DataFrame()
            a = temp_df[temp_df['Neuron']==N]
            one = a[a['Taste']==0]
            two = a[a['Taste']==1]
            three = a[a['Taste']==2]
            one['Trial'] = createList(0, len(one)-1)
            two['Trial'] = createList(0, len(two)-1)
            three['Trial'] = createList(0, len(three)-1)
            Temporary_df = pd.concat([one,two,three],axis=0).reset_index(drop=True)
            if N== 0:
                Temporary_df_all = Temporary_df
            else:
                Temporary_df_all = pd.concat([Temporary_df_all,Temporary_df],axis=0).reset_index(drop=True)
                
        test_pre,test_2_pre = ensemble_decoding(Temporary_df_all, ensemble_averaging=False, plot_confusion_matrix=False, cm_plot_title=None, n_trial_pairings=50, test_size=1/3, num_splits=20, class_labels=['Cold','Hot','Room'])
        dec_values_pre.append(test_pre.iloc[0,1])
    return dec_values_pre


