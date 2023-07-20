#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:00:09 2023

@author: robertovincis
"""


### Written by Cam Neese for the Vincis Lab ###

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt




"""
Data Accessing
"""

# This function returns one single spike train based on some identifying information. It requires a taste, neuron, and trial ID, and
# assumes you want the neuron data unless otherwise specified.

def get_spike_train(dataFrame, taste, neuron, trial, recording_type = 'Neuron'):
    
    start_index = dataFrame.columns.get_loc('Trial') + 1

    return dataFrame[(dataFrame['Recording Type'] == recording_type) & (dataFrame['Taste'] == taste)
                       & (dataFrame['Neuron'] == neuron) & (dataFrame['Trial'] == trial)].iloc[0,start_index:]

def get_spike_train_all(dataFrame, neuron, recording_type = 'Neuron'):
    
    start_index = dataFrame.columns.get_loc('Trial') + 1
    
    return dataFrame[(dataFrame['Recording Type'] == recording_type) 
                                         & (dataFrame['Neuron'] == neuron)]  

def get_min_trial_numb(dataFrame):

    min_trialN = min([dataFrame[(dataFrame['Taste'] == 0)]['Trial'].max(), 
                      dataFrame[(dataFrame['Taste'] == 1)]['Trial'].max(),
                      dataFrame[(dataFrame['Taste'] == 2)]['Trial'].max()])
    return min_trialN

"""
Data Editing
"""

# This function will take in the full-length spike trains and return subsections of them in another dataframe.
# This is written to accomodate common analyses we do. Below is a list of the 'result' parameter options and what they do.
# 'pre-taste': Returns recordings before and including taste administration. Used as a control. 
# 'post-taste': Returns recordings including and after taste administration. This is where the animal experiences the stimulus.
# 'one second': Returns recordings from the first second including and after taste administration. This is where the bulk of the processing #               is thought to happen.

def truncate(dataFrame, result = 'post-taste'):
    
    copy_data = dataFrame.copy()
    min_index = copy_data.columns.get_loc('Trial') + 1
    min_time = copy_data.columns[min_index]
    max_time = copy_data.columns[-1]
    max_index = copy_data.columns.get_loc(max_time)

    taste_index = int(np.floor((max_index + min_index) / 2))

    if result == 'pre-taste':
        copy_data.drop(copy_data.iloc[:, taste_index+1:], inplace=True, axis=1)
        return copy_data
    if result == 'post-taste':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index+1], inplace=True, axis=1)
        return copy_data
    if result == 'one second':
        post_taste = truncate(dataFrame,'post-taste')
        return truncate(post_taste, 'pre-taste')
    if result == '1.5 second':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index+1], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 1507:], inplace=True, axis=1)
        return copy_data
    if result == '0.5 second':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index+1], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 510:], inplace=True, axis=1)
        return copy_data
    if result == '0.1second':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index+1], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 110:], inplace=True, axis=1)
        return copy_data
    if result == '0.1-0.2second':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index+1], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 210:], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 9:110], inplace=True, axis=1)
        return copy_data
    if result == '0.2-0.3second':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index+1], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 310:], inplace=True, axis=1)
        copy_data.drop(copy_data.iloc[:, 9:210], inplace=True, axis=1)
        return copy_data


"""
Data smoothing
"""

# Smoothing a single spike train, likely taking in np.array or pd.series data.
# Code is based off of code found here:
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

def smooth_spike_train(x,window_len=100,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    edit: These additions will then be deleted

    input:
        x: the input signal as np array of size (4000,)
        window_len: the dimension of the smoothing window; should be an integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if window_len<3:
        return x


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    truncate_interval_length = int(np.floor(window_len/2))

    y = y[truncate_interval_length:-truncate_interval_length+1]
    
    if len(y) > len(x):
        y = y[:len(x)]

    return y


# Smoothing a collection of spike trains.

def smooth_all_spike_trains(dataFrame, window_len=100, window='hanning'):

    copy_data = dataFrame.copy()
    
    nSpikeTrains = copy_data.shape[0]
    start_index = copy_data.columns.get_loc('Trial') + 1
    start = copy_data.columns[start_index]
    end = copy_data.columns[-1] + 1
    
    spike_trains = np.array(copy_data.iloc[:,start_index:])
    
    for i in range(nSpikeTrains):
        spike_trains[i,:] = smooth_spike_train(spike_trains[i,:], window_len, window)
        
    smoothed_df = pd.DataFrame(spike_trains, columns = [j for j in range(start,end)])
    
    smoothed_df.insert(0,'Recording Type',np.array(copy_data['Recording Type']))
    smoothed_df.insert(1,'MouseID',np.array(copy_data['MouseID']))
    smoothed_df.insert(2,'Date',np.array(copy_data['Date']))
    smoothed_df.insert(3,'n_ID',np.array(copy_data['n_ID']))
    smoothed_df.insert(4,'Taste',np.array(copy_data['Taste'].astype(int)))
    smoothed_df.insert(5,'Neuron',np.array(copy_data['Neuron'].astype(int)))
    smoothed_df.insert(6,'Trial',np.array(copy_data['Trial'].astype(int)))

    return smoothed_df



"""
Data Subsampling
"""

def subsample_spike_train(spike_train, subsampling_rate=.1):

    """
    spike_train is a spike train of size (length,), with length = 2000 or 4000
    subsampling_rate is a percentage, the subsampled spike train will have size
    subsampling_rate*length, rounded
    """

    original_length = spike_train.shape[0]
    new_length = int(np.round(original_length*subsampling_rate))

    samples = np.round(np.linspace(1,original_length,new_length)).astype(int)-1

    subsampled_spike_train = spike_train[samples]

    return subsampled_spike_train


# Subsampling a smoothed spike train.

def subsample_all_spike_trains(dataFrame, subsampling_rate=.1):

    """
    subsampling_rate is a percentage, the subsampled spike trains will have size subsampling_rate*length, rounded
    """
    
    copy_data = dataFrame.copy()

    # The index of the column containing the first timestamp
    start_index = copy_data.columns.get_loc('Trial') + 1
    
    start_time = copy_data.columns[start_index]
    end_time = copy_data.columns[-1]
    original_length = end_time - start_time + 1
    
    new_length = int(np.round(original_length*subsampling_rate))
    
    samples = np.round(np.linspace(start_time+1,end_time,new_length)).astype(int)-1

    subsampled_df = copy_data[['Recording Type', 'Taste', 'Neuron', 'Trial'] + [j for j in samples]]

    return subsampled_df


"""
Data Binning
"""

# The number of spikes that occur over a given timespan are summed. 
# This function returns a binned representation of one spike train.

def bin_spike_train(spike_train, bin_width=100):
    
    original_length = spike_train.shape[0]
    
    # What happens if the length of the data doesn't evenly divide by the bin width?
    # The last bin will potentially be cut off.
    new_Dim = int(np.ceil(original_length/bin_width))
    
    binned = np.zeros(shape=new_Dim)
    for interval in range(new_Dim):
        int_start = interval * bin_width

        # If we've run out of room in the original spike train, set the end of the bin to be the end of the spike train
        if (interval+1) * bin_width > original_length:
            int_end = original_length
        else:
            int_end = (interval+1) * bin_width

#        binned[interval] = np.sum(spike_train[int_start:int_end])/bin_width 
        binned[interval] = np.sum(spike_train[int_start:int_end])
                
    return binned


def bin_all_spike_trains(dataFrame, bin_width=100):

    copy_data = dataFrame.copy()
    
    nSpikeTrains = copy_data.shape[0]
    start_index = copy_data.columns.get_loc('Trial') + 1
    start = copy_data.columns[start_index]
    end = copy_data.columns[-1] + 1
       
    spike_trains = np.array(copy_data.iloc[:,start_index:])
    binned_spike_trains = np.zeros(shape=(nSpikeTrains, int(np.ceil(spike_trains.shape[1]/bin_width))))
#    binned_spike_trains = np.zeros(shape=(nSpikeTrains, int(np.ceil(spike_trains.shape[1]))))
    
    for i in range(nSpikeTrains):
        binned_spike_trains[i,:] = bin_spike_train(spike_trains[i,:], bin_width)
        
    binned_df = pd.DataFrame(binned_spike_trains, columns = [j for j in range(start,end,bin_width)])
    
    binned_df.insert(0,'Recording Type',np.array(copy_data['Recording Type']))
    binned_df.insert(1,'MouseID',np.array(copy_data['MouseID']))
    binned_df.insert(2,'Date',np.array(copy_data['Date']))
    binned_df.insert(3,'n_ID',np.array(copy_data['n_ID']))
    binned_df.insert(4,'Taste',np.array(copy_data['Taste'].astype(int)))
    binned_df.insert(5,'Neuron',np.array(copy_data['Neuron'].astype(int)))
    binned_df.insert(6,'Trial',np.array(copy_data['Trial'].astype(int)))

    return binned_df


"""
Decoding
"""

# Support Vector Machines


# This function runs the SVM algorithm and returns one classifiction rate for one neuron. X is a numpy array (with each row as a spike train) and y is the taste labels.
""" NOTE: Generally speaking, avoid using this to do analysis. It's called by the other decoders. """
def SVM(X, y, confusion_mat=False, confusion_matrix_plot=False, cm_plot_title=None, class_labels=None, test_size = 1/3, num_splits=20):
    from sklearn.metrics import confusion_matrix
    # Which SVM Optimization problem do we solve?
    n_samples = X.shape[0] * (1-test_size) # Number of spike trains in the training set
    n_features = X.shape[1]  # Number of time points in the spike trains
    dual_param = (n_samples < n_features)

    # Define the SVM model
    model_SVM = LinearSVC(dual=dual_param, max_iter=10000, random_state=651)

    # Define arrays to record results
    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)
    split_scores = [] 

    for j in range(num_splits):                          # Use several splits of training and testing sets for robustness

        X_train,X_test,y_train,y_test = train_test_split(X,y,                 # This function is from sklearn
                                                         test_size = test_size, # Default: 2/3 of data to train and 1/3 to test
                                                         shuffle = True,
                                                         stratify = y)        # Sample from each taste

        model_SVM.fit(X_train,y_train)                   # Re-fit the classifier with the training set
        split_scores.append(model_SVM.score(X_test,y_test))  # Fit the testing set and record score
        
        if confusion_mat:
            y_true = np.concatenate((y_true, y_test)) # Record the 'true' taste
            y_pred = np.concatenate((y_pred, model_SVM.predict(X_test))) # Record the predicted taste
            cm = confusion_matrix(y_true,y_pred,normalize='true')
            
    if confusion_matrix_plot:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=class_labels,
            normalize='true',
            im_kw = {'vmin': 0.04, 'vmax': 0.3}
        )
        
        if cm_plot_title == None:
            cm_plot_title='Confusion Matrix'
        disp.ax_.set_title(cm_plot_title)
        disp.figure_.set_size_inches(10,10)
        plt.show()

    return split_scores, cm

# This function returns one classification rate per neuron in the dataframe it is given.

def single_neuron_decoding(dataFrame, confusion_mat=True, confusion_matrix_plot=False, cm_plot_title=None, class_labels=None, test_size = 1/3, num_splits=50, window_label=None): # add num_shuffle as variable
    
    if 'Trial' in dataFrame.columns:
        start_index = dataFrame.columns.get_loc('Trial') + 1
    else:
        start_index = 0
        
    neuron_list = dataFrame['Neuron'].unique()    
    # This will be the returned array, consisting of one classification rate per neuron
    mean_SVM_scores = []
    splits_SVM_scores = np.zeros(shape=(len(neuron_list),num_splits))
    cm_all = np.zeros(shape=(len(neuron_list),len(class_labels),len(class_labels)))

    # mean_SVM_scores_shuffl = np.zeros(shape=(len(neuron_list),num_shuffle))
    default_title = (cm_plot_title == None)
        

    # Iterate through all neurons
    for nindex, neuron in enumerate(neuron_list):
        neuron_df = dataFrame[dataFrame['Neuron']==neuron] # Select all spike trains from this neuron
        X = neuron_df.iloc[:,start_index:]  # X is the data. It has the shape (n_observations, n_times)
        y = neuron_df['Taste']                               # y is the labels. We're classifying based on taste.
        
        # Perform SVM
        if default_title:
            cm_plot_title=f'Confusion Matrix for Neuron {neuron}'
            if window_label != None:
                cm_plot_title = cm_plot_title + window_label
                
        svm_scores, cm = SVM(X, y, confusion_mat=confusion_mat, confusion_matrix_plot=confusion_matrix_plot, cm_plot_title=cm_plot_title, class_labels=class_labels, test_size=test_size, num_splits=num_splits)
        
        mean_SVM_scores.append(np.mean(svm_scores))
        splits_SVM_scores[nindex, :] = svm_scores
        cm_all[nindex,:,:] = cm
        
    # Make the dataframes to return
    mean_score_dict = {'Neuron': neuron_list, 'Overall SVM Score': mean_SVM_scores}
    mean_score_df = pd.DataFrame(mean_score_dict)
    
    split_score_df = pd.DataFrame(splits_SVM_scores, columns=range(num_splits))
    split_score_df.insert(0, 'Neuron', neuron_list)
                               
                               
    return mean_score_df, split_score_df, cm_all

# This function returns one "classification-score" per neuron in the dataframe it is given + the classification score after the labels have been shuffled 10 times. The 95% of the "suffled-classification-score" is then quantified. If the 95% of the "suffled-classification-score" is lower than the "classification-score" then the neuron is deemed coding neuron

def single_neuron_decoding_plus_shuffl(dataFrame, class_labels=None):
    
    import numpy as np
    from tqdm import tqdm
    snd_mean_scores, snd_splits_scores, cm_all = single_neuron_decoding(dataFrame, 
                                                                confusion_mat=True, 
                                                                confusion_matrix_plot=False,
                                                                class_labels=class_labels)
    neuron_list = dataFrame['Neuron'].unique()
    snd_mean_scores.index = neuron_list
    snd_splits_scores.index = neuron_list
    snd_mean_scores_shuffl = pd.DataFrame()
    cm_all_sh_big = np.zeros(shape=(100,len(neuron_list),len(class_labels),len(class_labels)))

    for nindex, neuron in tqdm(enumerate(neuron_list)):
        temp_df = dataFrame.loc[dataFrame['Neuron']==neuron]
        temp = pd.DataFrame()
        for j in range (100):
            lab = np.array(temp_df['Taste'])
            np.random.shuffle(lab)
            test_df_shuffled = temp_df.copy()
            test_df_shuffled.loc[:,'Taste'] = lab    
            snd_mean_scores_sh, snd_splits_scores_sh, cm_all_sh = single_neuron_decoding(test_df_shuffled, 
                                                                                         confusion_mat=True, 
                                                                                         confusion_matrix_plot=False,
                                                                                         class_labels=class_labels)   
            cm_all_sh_big[j,nindex,:,:]=cm_all_sh  
            a = snd_mean_scores_sh['Overall SVM Score'][0]
    #        temp.at[j,'Overall SVM Score'] = a
            temp.at[neuron,'SVM Score shuffl_'+ str(j)] = a

        snd_mean_scores_shuffl = pd.concat((snd_mean_scores_shuffl,temp),axis=0)
        
    snd_mean_scores_shuffl.T.describe(percentiles=[.95])
    snd_mean_scores_shuffl.quantile(q=0.95, axis= 'columns')
    snd_mean_scores['95% shuffle'] = snd_mean_scores_shuffl.quantile(q=0.95, axis= 'columns')

    snd_mean_scores_copy = snd_mean_scores.copy()

    snd_mean_scores_copy['diff'] = snd_mean_scores_copy['Overall SVM Score']-snd_mean_scores_copy['95% shuffle']
    snd_mean_scores_copy['rank'] = snd_mean_scores_copy['diff'].rank(ascending=False)
    if class_labels == ['C','N','Q','S']:
        snd_mean_scores_copy['flag_Taste'] = snd_mean_scores_copy['Overall SVM Score']>snd_mean_scores_copy['95% shuffle']
    elif class_labels == ['C','H']:
        snd_mean_scores_copy['flag_Temp'] = snd_mean_scores_copy['Overall SVM Score']>snd_mean_scores_copy['95% shuffle']
    elif class_labels == ['Cold','Hot','Room']:
        snd_mean_scores_copy['flag_Temp'] = snd_mean_scores_copy['Overall SVM Score']>snd_mean_scores_copy['95% shuffle']

        
#TODO add other conditions here to make sure the pandas dataframe columns names are different
            
    fig, ax = plt.subplots()
    ax.plot(snd_mean_scores_copy['rank'],snd_mean_scores_copy['Overall SVM Score'],'or',label='SVM Score',markersize=7)
    ax.plot(snd_mean_scores_copy['rank'],snd_mean_scores_copy['95% shuffle'],'ob',label='SVM Score Shuffled labels',markersize=5, alpha=0.2)
#TODO add other conditions here to make sure the pandas dataframe columns names are different
    if class_labels == ['C','N','Q','S']:
        ax.plot(snd_mean_scores_copy['rank'],snd_mean_scores_copy['flag_Taste'],'xb',label='SVM Score Shuffled labels')
    elif class_labels == ['C','H']:
        ax.plot(snd_mean_scores_copy['rank'],snd_mean_scores_copy['flag_Temp'],'xb',label='SVM Score Shuffled labels')
    elif class_labels == ['Cold','Hot','Room']:
        ax.plot(snd_mean_scores_copy['rank'],snd_mean_scores_copy['flag_Temp'],'xb',label='SVM Score Shuffled labels')

    ax.legend()
    ax.set_ylabel('Classification score')
    ax.set_xlabel('Water responsive neuron ID')
    ax.set_title('SVM for temperature')

    return snd_mean_scores_shuffl,snd_mean_scores_copy,cm_all,cm_all_sh_big

    
# This function uses all neurons in the dataset to make a single classification score.

# If ensemble_averaging is set to True, the average SVM score will be taken. That is, SVM will be run independently on each neuron,
# and the results averaged.

# If ensemble_averaging is set to False, a very advanced algorithm will sample spike trains and concatenate them all into 

def ensemble_decoding(dataFrame, ensemble_averaging=False, plot_confusion_matrix=False, cm_plot_title=None, n_trial_pairings=50, test_size=1/3, num_splits=20, class_labels=None):        
    
    # Get some helpful information  
    if 'Trial' in dataFrame.columns:
        start_index = dataFrame.columns.get_loc('Trial') + 1
    else:
        start_index = 0

    data_len = dataFrame.iloc[:,start_index:].shape[1]
    nNeurons = dataFrame['Neuron'].nunique()
    tastes = dataFrame['Taste'].unique()
    neurons = dataFrame['Neuron'].unique()
    
    # If ensemble_averaging is set to True, then we just average the individual SVM scores for each neuron.
    if ensemble_averaging == True:
        
        mean_SVM_scores = []
        splits_SVM_scores = np.zeros(shape=(nNeurons, num_splits))
        # Keep track of all true and predicted test labels.
        y_true = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)
        
        for nindex, neuron in enumerate(neurons):
            neuron_df = dataFrame[dataFrame['Neuron']==neuron] # Select all spike trains from this neuron
            X = neuron_df.iloc[:,start_index:]  # X is the data. It has the shape (n_observations, n_times)
            y = neuron_df['Taste']                               # y is the labels. We're classifying based on taste.
        
            # Which SVM Optimization problem do we solve?
            n_samples = X.shape[0] * (1-test_size) # Number of spike trains in the training set
            n_features = X.shape[1]  # Number of time points in the spike trains
            dual_param = (n_samples < n_features)

            # Define the SVM model
            model_SVM = LinearSVC(dual=dual_param, max_iter=10000, random_state=651)

            # Define arrays to record results
            split_scores = [] 

            for j in range(num_splits):                          # Use several splits of training and testing sets for robustness

                X_train,X_test,y_train,y_test = train_test_split(X,y,                 # This function is from sklearn
                                                                 test_size = test_size, # Default: 2/3 of data to train and 1/3 to test
                                                                 shuffle = True,
                                                                 stratify = y)        # Sample from each taste

                model_SVM.fit(X_train,y_train)                   # Re-fit the classifier with the training set
                split_scores.append(model_SVM.score(X_test,y_test))  # Fit the testing set and record score

                if plot_confusion_matrix:
                    y_true = np.concatenate((y_true, y_test)) # Record the 'true' taste
                    y_pred = np.concatenate((y_pred, model_SVM.predict(X_test))) # Record the predicted taste
            
            # Record the average of the intra-neuron splits
            mean_SVM_scores.append(np.mean(split_scores))
            splits_SVM_scores[nindex, :] = split_scores
            
            
        if plot_confusion_matrix:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=class_labels,
                normalize='true',
                im_kw = {'vmin': 0, 'vmax': 1}
            )
            if cm_plot_title == None:
                cm_plot_title=f'Ensemble Averaged across {nNeurons} Neurons'
            disp.ax_.set_title(cm_plot_title)
            disp.figure_.set_size_inches(10,10)
            plt.show()
            
        # Set up the dataframes to return
        mean_score_dict = {'Neuron': f'Average of {nNeurons} Neurons', ' Overall SVM Score': np.mean(mean_SVM_scores)}
        mean_score_df = pd.DataFrame(mean_score_dict, index=[0])

        split_score_df = pd.DataFrame(splits_SVM_scores, columns=range(num_splits))
        split_score_df.insert(0, 'Neuron', neurons)

        return mean_score_df, split_score_df
        
    else:
        
        # Figure out the minimum number of trials a neuron has for each taste
        taste_mins = []
        
        for taste in tastes:
            taste_trials = []
            for neuron in neurons:
                ntdf = dataFrame[(dataFrame['Neuron'] == neuron) & (dataFrame['Taste'] == taste)]
                taste_trials.append(ntdf.shape[0])
            taste_mins.append(min(taste_trials))
        
        
        
        # Set up arrays for recording results
        y_true = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)
        split_scores = [] 
        pairing_split_scores = np.zeros(shape=(n_trial_pairings,num_splits))
        
        # Repeat sampling of trials
        for T in range(n_trial_pairings):
            
            y = np.repeat(tastes, taste_mins)
            # X is the concatenated dataframe. Each row will represent a trial, and the neural spike trains will be stacked horizonatally.
            X = np.zeros(shape=(len(y), nNeurons*data_len))
        
            # Iterate through each neuron
            for nindex, neuron in enumerate(neurons):
                trialindex = 0
                # Iterate through each taste and select the appropriate number of trials
                for tindex, taste in enumerate(tastes):
                    ntdf = dataFrame[(dataFrame['Neuron'] == neuron) & (dataFrame['Taste'] == taste)]
                    trials = np.array(ntdf['Trial'])
                    selected_trials = np.random.choice(trials, taste_mins[tindex], replace=False)
                    for trial in selected_trials:
                        # Put it into X
                        X[trialindex, (nindex*data_len):(nindex*data_len)+data_len] = np.array(ntdf[ntdf['Trial'] == trial].iloc[:,start_index:])
                        trialindex += 1
                        
            # Now we have our concatenated data and labels. Let's run SVM.
            
            n_samples = X.shape[0] * (1-test_size) # Number of spike trains in the training set
            n_features = X.shape[1]  # Number of time points in the spike trains
            dual_param = (n_samples < n_features)

            # Define the SVM model
            model_SVM = LinearSVC(dual=dual_param, max_iter=10000, random_state=651)
            
            for j in range(num_splits):                          # Use several splits of training and testing sets for robustness

                X_train,X_test,y_train,y_test = train_test_split(X,y,                 # This function is from sklearn
                                                                 test_size = test_size, # Default: 2/3 of data to train and 1/3 to test
                                                                 shuffle = True,
                                                                 stratify = y)        # Sample from each taste

                model_SVM.fit(X_train,y_train)                   # Re-fit the classifier with the training set
                s_score = model_SVM.score(X_test,y_test) # Fit the testing set
                split_scores.append(s_score)   # record score
                pairing_split_scores[T, j] = s_score
                
                if plot_confusion_matrix:
                    y_true = np.concatenate((y_true, y_test)) # Record the 'true' taste
                    y_pred = np.concatenate((y_pred, model_SVM.predict(X_test))) # Record the predicted taste
                    
        # Plot the confusion matrix if applicable
        if plot_confusion_matrix:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=class_labels,
                normalize='true',
                im_kw = {'vmin': 0, 'vmax': 1}
            )
            
            if cm_plot_title == None:
                cm_plot_title=f'Ensemble of {nNeurons} Neurons'
            disp.ax_.set_title(cm_plot_title)
            disp.figure_.set_size_inches(10,10)
            plt.show()
            
            
            
        # Make the dataframe to return
        mean_SVM_dict = {'Neuron': f'Ensemble of {nNeurons} Neurons', 'Overall SVM Score': np.mean(split_scores)}
        mean_SVM_df = pd.DataFrame(mean_SVM_dict, index=[0])

        splits_SVM_df = pd.DataFrame(pairing_split_scores, columns=range(num_splits))
        splits_SVM_df.insert(0, 'Trial Pairing', range(n_trial_pairings))
        
        
        return mean_SVM_df, splits_SVM_df
    
    
    
    
# This function chops up the full signal into window_size-length pieces. You must specify a method: 'ensemble' or 'single' for the decoding type.
    
def sliding_window_decoding(dataFrame, method=None, ensemble_averaging=False, window_size=None, plot_confusion_matrix=False, n_trial_pairings=50, class_labels=None, test_size=1/3, num_splits=20):
    
    # Deal with parameter selection issues
    if method not in ['ensemble', 'single']:
        raise ValueError("Must specify a method. Set method parameter to 'single' for single-neuron decoding and 'ensemble' for ensemble decoding.")
    if method == 'ensemble':
        print(f'Proceeding with ensemble_averaging set to {ensemble_averaging}.')
        
        
    # This function assumes that the data starts in the column after 'Trials'
    if 'Trial' in dataFrame.columns:
        start_index = dataFrame.columns.get_loc('Trial') + 1
    else:
        start_index = 0
        
    data_len = dataFrame.iloc[:,start_index:].shape[1]
    nNeurons = dataFrame['Neuron'].nunique()
    
    # Figure out bins    
    if window_size==None:
        print("No window size specified. Proceeding using full available spike train.")
        nBins=1
        window_size=data_len
    else:
        # Determine number of bins
        nBins = int(np.floor(data_len/window_size))
        
    bin_names = []
        
    # Make an array to store the SVM scores in
    if method == 'ensemble':
        # Using all neurons together, so we need just one row
        mean_scores_array = np.zeros(shape=(1,nBins))
    elif method == 'single':
        mean_scores_array = np.zeros(shape=(nNeurons, nBins))

    splits_frames = []    
        
    
    for b in range(nBins):
        index_1 = start_index + (b*window_size)
        index_2 = start_index + (b*window_size) + window_size - 1

        # Keep track of the start and end of the bins used
        bin_names.append(f'{dataFrame.columns[index_1]} to {dataFrame.columns[index_2]}')

        # Pull off only the data from this bin
        spike_train_segments = np.array(dataFrame.iloc[:, index_1:index_2])

        # Re-attach the identifying information
        window_df = pd.DataFrame(spike_train_segments)
        window_df.insert(0, 'Taste', dataFrame['Taste'])
        window_df.insert(1, 'Neuron', dataFrame['Neuron'])
        window_df.insert(2, 'Trial', dataFrame['Trial'])

        if method == 'ensemble':

            if ensemble_averaging == True:
                cm_plot_title = f'Confusion Matrix for Average of {nNeurons} Neurons \n Window {bin_names[-1]}'
            elif ensemble_averaging == False:
                cm_plot_title = f'Confusion Matrix for Ensemble of {nNeurons} Neurons \n Window {bin_names[-1]}'   
                
            # Perform Ensemble Averaged SVM
            mean_scores, splits_scores = ensemble_decoding(window_df, ensemble_averaging=ensemble_averaging, plot_confusion_matrix=plot_confusion_matrix, cm_plot_title=cm_plot_title, n_trial_pairings=n_trial_pairings, test_size=test_size, num_splits=num_splits, class_labels=class_labels)

            # Store information for the means
            mean_scores_array[0,b] = np.array(mean_scores['Overall SVM Score'])
            neuron_list = np.array(mean_scores['Neuron'])

            # Store information for the splits
            splits_scores.insert(0,'Window', bin_names[-1])
            splits_frames.append(splits_scores)


        elif method == 'single':
            
            mean_scores, splits_scores = single_neuron_decoding(window_df, plot_confusion_matrix=plot_confusion_matrix, test_size=test_size, num_splits=num_splits, window_label = f' \n Window {bin_names[-1]}', class_labels=class_labels)
            
            # Store information for the means
            mean_scores_array[:,b] = np.array(mean_scores['Overall SVM Score'])
            neuron_list = np.array(mean_scores['Neuron'])
            
            # Store information for the splits
            splits_scores.insert(0,'Window', bin_names[-1])
            splits_frames.append(splits_scores)

    # Make the data pretty
    mean_SVM_df = pd.DataFrame(mean_scores_array, columns=bin_names)
    mean_SVM_df.insert(0, 'Neuron', neuron_list)
    
    splits_SVM_df = pd.concat(splits_frames, ignore_index=True)
    
    
    return mean_SVM_df, splits_SVM_df
    


def plot_raster_psth(dataFrame_r, dataFrame_p, neuron, bin_width, start, end, class_labels = ['Cold', 'Room', 'Hot'], plot_all=False):
    
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    
    curr_n_ras = get_spike_train_all(dataFrame_r, neuron)
    curr_n = get_spike_train_all(dataFrame_p, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1
    col = ['blue','black','red'] # color list for temp
    col_t = ['darkgreen','crimson','indigo','maroon','tomato'] # color list for taste
    
    unit_all = []
    for tt in range(len(class_labels)): # change it to lenght of class_labels
        rast = curr_n_ras[curr_n_ras['Taste']== tt].iloc[0:min_trialN,start_index:].reset_index(drop=True).T
        #rast.loc[rast.loc[0]]==1
        unit=[]
        for t in range(rast.shape[1]):
            temp = []
            for j in range(-2000, 2000, 1):
                if rast[t][j]>0:
                    temp.append(j)
            unit.append(temp)
        unit_all.append(unit)
    if len(class_labels)==3:
        s_0 = pd.Series(unit_all[0])
        s_1 = pd.Series(unit_all[1])
        s_2 = pd.Series(unit_all[2])
        s = pd.Series(pd.concat((s_0,s_1,s_2),axis=0))
    else:
        s_0 = pd.Series(unit_all[0])
        s_1 = pd.Series(unit_all[1])
        s_2 = pd.Series(unit_all[2])
        s_3 = pd.Series(unit_all[3])
        s_4 = pd.Series(unit_all[4])
        s = pd.Series(pd.concat((s_0,s_1,s_2,s_3,s_4),axis=0))

    fig = plt.figure()
    gs0 = gridspec.GridSpec(5, 2, figure=fig)
    ax1 = fig.add_subplot(gs0[3:5,0:2])
    if not plot_all:
            for t in range(len(class_labels)): # change it to len(class_labels)
                if len(class_labels)==3:
                    ax1.plot((curr_n[curr_n['Taste']== t].iloc[0:min_trialN,start_index:].T).mean(axis=1)/(bin_width/1000), label = class_labels[t],color = col[t])
                else:
                    ax1.plot((curr_n[curr_n['Taste']== t].iloc[0:min_trialN,start_index:].T).mean(axis=1)/(bin_width/1000), label = class_labels[t],color = col_t[t])
                ax1.legend()
                ax1.set_xlim(-start,end)
                ax1.set_xlabel('Time (ms)')
                ax1.set_ylabel('Firing rate')
    else:
        ax1.plot((curr_n.iloc[:,start_index:].T).mean(axis=1)/(bin_width/1000), label = 'all trials', color = 'k')
        ax1.legend()
        ax1.set_xlim(-start,end)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Firing rate')

    ax2 = fig.add_subplot(gs0[0:3,0:2])
    if plot_all:
        plt.eventplot(s,color='black')
    else:
        if len(class_labels)==3:
            plt.eventplot(s,color=[col[0]]*min_trialN + [col[1]]*min_trialN + [col[2]]*min_trialN)
        else:
            plt.eventplot(s,color=[col_t[0]]*min_trialN + 
                                  [col_t[1]]*min_trialN + 
                                  [col_t[2]]*min_trialN + [col_t[3]]*min_trialN + [col_t[4]]*min_trialN)
    ax2.set_xlim(-start,end)
    ax2.spines.bottom.set_visible(False) 
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False) 
    ax2.spines.left.set_visible(False) 
    ax2.axes.yaxis.set_ticks([])
    ax2.axes.xaxis.set_ticks([])
    #ax2.set_xlabel('Time (s)',fontdict=font)
    ax2.set_ylabel('Trials')
    neuronID = dataFrame_r.loc[dataFrame_r['Neuron']==neuron]['n_ID'].reset_index(drop=True)[0][:-4]
    mouse = dataFrame_r.loc[dataFrame_r['Neuron']==neuron]['MouseID'].reset_index(drop=True)[0]
    date = dataFrame_r.loc[dataFrame_r['Neuron']==neuron]['Date'].reset_index(drop=True)[0]
    ax2.set_title(mouse + '_' + date + '_' + neuronID)
    
def plot_raster_psth_smooth(dataFrame_r, dataFrame_s, neuron, start, end, class_labels = ['Cold', 'Room', 'Hot'],plot_all=False):
    
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300  
    
    curr_n_ras = get_spike_train_all(dataFrame_r, neuron)
    curr_n = get_spike_train_all(dataFrame_s, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1
    col = ['blue','black','red']
    col_t = ['darkgreen','crimson','indigo','maroon','tomato'] # color list for taste
    
    unit_all = []
    for tt in range(len(class_labels)): # change it to lenght of class_labels
        rast = curr_n_ras[curr_n_ras['Taste']== tt].iloc[0:min_trialN,start_index:].reset_index(drop=True).T
        #rast.loc[rast.loc[0]]==1
        unit=[]
        for t in range(rast.shape[1]):
            temp = []
            for j in range(-2000, 2000, 1):
                if rast[t][j]>0:
                    temp.append(j)
            unit.append(temp)
        unit_all.append(unit)
    if len(class_labels)==3:
        s_0 = pd.Series(unit_all[0])
        s_1 = pd.Series(unit_all[1])
        s_2 = pd.Series(unit_all[2])
        s = pd.Series(pd.concat((s_0,s_1,s_2),axis=0))
    else:
        s_0 = pd.Series(unit_all[0])
        s_1 = pd.Series(unit_all[1])
        s_2 = pd.Series(unit_all[2])
        s_3 = pd.Series(unit_all[3])
        s_4 = pd.Series(unit_all[4])
        s = pd.Series(pd.concat((s_0,s_1,s_2,s_3,s_4),axis=0))

    min_index = dataFrame_s.columns.get_loc('Trial') + 1
    copy_data = dataFrame_s.copy()

    copy_data_N = copy_data[copy_data['Neuron']==neuron]
    if len(class_labels)==3:
        copy_data_N_0 = copy_data_N[copy_data_N['Taste']==0]
        copy_data_N_1 = copy_data_N[copy_data_N['Taste']==1]
        copy_data_N_2 = copy_data_N[copy_data_N['Taste']==2]

        copy_data_N_0_C = copy_data_N_0.copy() 
        copy_data_N_0_C.drop(copy_data_N_0_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_1_C = copy_data_N_1.copy() 
        copy_data_N_1_C.drop(copy_data_N_1_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_2_C = copy_data_N_2.copy() 
        copy_data_N_2_C.drop(copy_data_N_2_C.iloc[:, :min_index], inplace=True, axis=1)
    else:
        copy_data_N_0 = copy_data_N[copy_data_N['Taste']==0]
        copy_data_N_1 = copy_data_N[copy_data_N['Taste']==1]
        copy_data_N_2 = copy_data_N[copy_data_N['Taste']==2]
        copy_data_N_3 = copy_data_N[copy_data_N['Taste']==3]
        copy_data_N_4 = copy_data_N[copy_data_N['Taste']==4]        
        
        copy_data_N_0_C = copy_data_N_0.copy() 
        copy_data_N_0_C.drop(copy_data_N_0_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_1_C = copy_data_N_1.copy() 
        copy_data_N_1_C.drop(copy_data_N_1_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_2_C = copy_data_N_2.copy() 
        copy_data_N_2_C.drop(copy_data_N_2_C.iloc[:, :min_index], inplace=True, axis=1)
        
        copy_data_N_3_C = copy_data_N_3.copy() 
        copy_data_N_3_C.drop(copy_data_N_3_C.iloc[:, :min_index], inplace=True, axis=1)
        
        copy_data_N_4_C = copy_data_N_4.copy() 
        copy_data_N_4_C.drop(copy_data_N_4_C.iloc[:, :min_index], inplace=True, axis=1)

    plt.xlabel = ('Time (ms)')
    plt.ylabel = ('Firing Rate')
    fig = plt.figure()
    gs0 = gridspec.GridSpec(5, 2, figure=fig)
    ax1 = fig.add_subplot(gs0[3:5,0:2])
    if not plot_all:
        if len(class_labels)==3:
            ax1.plot(copy_data_N_2_C.mean()*1000,color = 'red',label = 'Hot')
            ax1.plot(copy_data_N_0_C.mean()*1000,color = 'blue',label = 'Cold')
            ax1.plot(copy_data_N_1_C.mean()*1000,color = 'black',label = 'Room')
        else:
            ax1.plot(copy_data_N_0_C.mean()*1000,color = col_t[0],label = 'Water')
            ax1.plot(copy_data_N_1_C.mean()*1000,color = col_t[1],label = 'NaCl')
            ax1.plot(copy_data_N_2_C.mean()*1000,color = col_t[2],label = 'Citric acid')
            ax1.plot(copy_data_N_3_C.mean()*1000,color = col_t[3],label = 'Quinine')
            ax1.plot(copy_data_N_4_C.mean()*1000,color = col_t[4],label = 'Sucrose')
        ax1.legend()
    else:
        copy_data_N_plot = copy_data_N.copy()
        copy_data_N_plot.drop(copy_data_N.iloc[:, :min_index], inplace=True, axis=1)
        ax1.plot(copy_data_N_plot.mean()*1000,color = 'black',label = 'all trials')
        ax1.legend()
    ax1.set_xlim(-start,end)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Firing rate')

    ax2 = fig.add_subplot(gs0[0:3,0:2])
    if plot_all:
        plt.eventplot(s,color='black')
    else:
        if len(class_labels)==3:            
            plt.eventplot(s,color=[col[0]]*min_trialN + [col[1]]*min_trialN + [col[2]]*min_trialN)
        else:
            plt.eventplot(s,color=[col_t[0]]*min_trialN + [col_t[1]]*min_trialN + [col_t[2]]*min_trialN + 
                                  [col_t[3]]*min_trialN + [col_t[4]]*min_trialN )
    ax2.set_xlim(-start,end)
    ax2.spines.bottom.set_visible(False) 
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False) 
    ax2.spines.left.set_visible(False) 
    ax2.axes.yaxis.set_ticks([])
    ax2.axes.xaxis.set_ticks([])
    #ax2.set_xlabel('Time (s)',fontdict=font)
    ax2.set_ylabel('Trials')
    neuronID = dataFrame_r.loc[dataFrame_r['Neuron']==neuron]['n_ID'].reset_index(drop=True)[0][:-4]
    mouse = dataFrame_r.loc[dataFrame_r['Neuron']==neuron]['MouseID'].reset_index(drop=True)[0]
    date = dataFrame_r.loc[dataFrame_r['Neuron']==neuron]['Date'].reset_index(drop=True)[0]
    plt.title(mouse + '_' + date + '_' + neuronID)
    
def newplot(dataframe,filePath,mouse,Date,N=[0],start=500,end=2000,
            bin_width=200,smoothWin=300,plot_all=False,class_labels = ['Cold','Room','Hot']):
    """Summary or Description of the Function
     
     Creator: Roberto Vincis, FSU - 090222
     
     Parameters:
     dataframe (pandas): pandas dataframe containing all the spikes information in {0,1} format.
     filePath (str): current working directory.
     mouse (str): id of the mouse.
     Date (str): id of the date.
     N (list): is of the neurons you want to analyze.
     start (int): start point of the rasters and PSTHs in millisecond.
     end (int): end point of the rasters and PSTHs in millisecond.
     bin_width: binning windows in milliseconds.
     smoothWin (int): smoothing window value in milliseconds.
     plot_all (bool): True if you want to plot all stimuli combined together - False if you want to split them
     
     Returns:
         
     Dependencies:
     This function call other functions:
         1 - smooth_all_spike_trains
         2 - bin_all_spike_trains
         3 - plot_raster_psth
         4 - plot_raster_psth_smooth
    """
    import os
    import platform
    import matplotlib.pyplot as plt
    from importlib import reload
    plt=reload(plt)
    
    for n in N:
        #class_labels = ['Cold', 'Room', 'Hot']
        neuron_df = dataframe[(dataframe['Recording Type']=='Neuron')] # Select all spike trains from this neuron
        neuron_df_smooth = smooth_all_spike_trains(neuron_df, smoothWin)
        neuron_df_binned = bin_all_spike_trains(neuron_df, bin_width)
        plot_raster_psth(neuron_df, neuron_df_binned, n, bin_width, start, end,class_labels = class_labels,plot_all=plot_all) # plot raster and psth binned
        #m = neuron_df['MouseID'][neuron_df['Neuron']==n].reset_index(drop=True)[0]
        nid = neuron_df['n_ID'][neuron_df['Neuron']==n].reset_index(drop=True)[0][:-4]
        if plot_all:
            neuronid = str(nid + "_all_binned.pdf")
        else:
            neuronid = str(nid + "_binned.pdf")
        if platform.system() == 'Darwin':
            fileNameFig =  str(str(filePath) + '/' + mouse +  '/' + Date + '/SU_Figures/')
            if not os.path.exists(fileNameFig):
                os.makedirs(fileNameFig)
        else:
            fileNameFig =  str(str(filePath) + '\\' + mouse +  '\\' + Date + '\\SU_Figures\\')
            if not os.path.exists(fileNameFig):
                os.makedirs(fileNameFig)
                
        plt.savefig(fileNameFig + neuronid,bbox_inches='tight')

        plot_raster_psth_smooth(neuron_df, neuron_df_smooth, n, start, end,class_labels = class_labels,plot_all=plot_all) # plot raster anpsth smoothed
        if plot_all:
            neuronid = str(nid + "_all_smoothed.pdf")
        else:
            neuronid = str(nid + "_smoothed.pdf")
            
        if platform.system() == 'Darwin':
            fileNameFig =  str(str(filePath) + '/' + mouse +  '/' + Date + '/SU_Figures/')
            if not os.path.exists(fileNameFig):
                os.makedirs(fileNameFig)
        else:
            fileNameFig =  str(str(filePath) + '\\' + mouse +  '\\' + Date + '\\SU_Figures\\')
            if not os.path.exists(fileNameFig):
                os.makedirs(fileNameFig)
                
        plt.savefig(fileNameFig + neuronid,bbox_inches='tight')
        
def extr_responsivness_all(dataframe,filePath,mouse,Date, N=[0],smoothWin=300,Basel= -750):
     """Summary or Description of the Function
     
     Creator: Roberto Vincis, FSU - 090222
     
     Parameters:
     dataframe (pandas): pandas dataframe containing all the spikes information in {0,1} format.
     filePath (str): current working directory.
     mouse (str): id of the mouse.
     Date (str): id of the date.
     N (list): is of the neurons you want to analyze
     smoothWin (int): smoothing window value in milliseconds
     Basel (int): time windows of the baseline (before time 0) in milliseconds (note this has to be negative)
     
     Returns:
     updated pandas dataframe: Sum_Long.

     """
     final_Folder = 'SU_Analysis/Sum_Long.csv'
     Sum_Long = pd.read_csv(filePath / mouse / Date / final_Folder)
     Sum_Long = Sum_Long.drop(columns=['Unnamed: 0'])
     Lat = list()
     Dur = list()
     Si = list()
     neuron_df = dataframe[(dataframe['Recording Type']=='Neuron')] # Select all spike trains from this neuron
     neuron_df_smooth = smooth_all_spike_trains(neuron_df, smoothWin)
     for n in N:
         copy_data_N = neuron_df_smooth[neuron_df_smooth['Neuron']==n]
         copy_data_N_1 = copy_data_N.copy()
         min_index = copy_data_N.columns.get_loc(Basel)
         copy_data_N_1.drop(copy_data_N_1.iloc[:, :min_index], inplace=True, axis=1)

         baseline = copy_data_N_1.copy()
         evoked = copy_data_N_1.copy()

         min_index = baseline.columns.get_loc(0)
         baseline.drop(baseline.iloc[:, min_index:], inplace=True, axis=1)
         evoked.drop(baseline.iloc[:, :min_index], inplace=True, axis=1)

         M = baseline.mean().mean()
         SD_q = baseline.std().quantile(q=0.95)
         upper_n = M+SD_q
         lower_n = M-SD_q
         M_ev = evoked.mean()
         if (baseline.mean().mean()<1e-04) and (evoked.mean().mean()<1e-04):
             sign = 0
             latency = None
             duration = None
             Lat.append(latency)
             Dur.append(duration)
             Si.append(sign) 
         else:
             
             if M_ev[M_ev>upper_n].any()or M_ev[M_ev<lower_n].any():
                 lat_0  = M_ev.loc[pd.Index(M_ev>upper_n)]
                 lat_1  = M_ev.loc[pd.Index(M_ev<lower_n)]
                 latency = pd.concat((lat_0,lat_1)).index[0]/1000
                 duration = pd.concat((lat_0,lat_1)).shape[0]
                 if lat_0.empty:
                     sign = -1
                 elif lat_1.empty:
                     sign = 1
                 else:
                     if lat_0.index[0]<lat_1.index[0]:
                         sign = 1
                     else:
                         sign = -1
                 Lat.append(latency)
                 Dur.append(duration)
                 Si.append(sign)
             else:
                 sign = 0
                 latency = None
                 duration = None
                 Lat.append(latency)
                 Dur.append(duration)
                 Si.append(sign)

         fig,ax = plt.subplots()
         plt.plot(copy_data_N_1.mean())
         plt.plot(M_ev)         
         plt.hlines(y=M,xmin = -500, xmax = 2000,color = 'green')
         plt.hlines(y=upper_n,xmin = -500, xmax = 2000,color = 'pink')
         plt.hlines(y=lower_n,xmin = -500, xmax = 2000,color = 'pink')

         if sign != 0:
             plt.text(1500,copy_data_N_1.mean().max(),'Resp.= Yes!')
             plt.text(1500,(copy_data_N_1.mean().max())-0.1*copy_data_N_1.mean().max(),'Latency= ' + str(latency) +' ms')
             plt.text(1500,(copy_data_N_1.mean().max())-0.2*copy_data_N_1.mean().max(),'Duration= ' + str(duration) +' ms')
         else:
             plt.text(1500,copy_data_N_1.mean().max(),'Resp.= No!')
         m = mouse
         nid = Sum_Long['name Wav'][n][:-4]
         neuronid = m + '_SU_' + str(nid) + '.pdf'
         plt.text(0,copy_data_N_1.mean().max(),m + '_SU_' + str(nid))
         plt.savefig(str(filePath / mouse / Date / final_Folder[0:11]) + '/' + neuronid,bbox_inches='tight')

     Sum_Long['Resp_all'] = Si
     Sum_Long['Latency'] = Lat
     Sum_Long['Duration'] = Dur
     Sum_Long.to_csv(filePath / mouse / Date / final_Folder)
     return Sum_Long
        
"""

This is the code that would prepare the dataframe containing all spikes
Dataframe

"""

def prepare_datafr_spikes(filePath,mouse,Date):
    """Summary or Description of the Function
     
     Creator: Roberto Vincis, FSU - 090222
     
     Parameters:
     filePath (str): current working directory.
     mouse (str): id of the mouse.
     Date (str): id of the date.
     
     Returns:
     updated pandas dataframe: allN.
         rows dataframe:
         columns dataframe:

    """
#    from pathlib import Path
    import numpy as np
    import pickle
    import pandas as pd
    from tqdm import tqdm
    import os
    import re
    from pathlib import Path  
    
    # 0 - input to the function
    #filePath = Path('/Users/robertovincis') # set the base folder
    #mouse = 'CB303' # ID of mouse to analyze
    #Date = '061622' # date 
    
    
    # 1 - import the dataframe with id of all sorted neuron for the current exp. session -----------------------------
    nameFile = 'SU_Analysis/Sum_Long.csv'
    file_to_load= filePath / mouse / Date / nameFile
    df = pd.read_csv(file_to_load,index_col=0)
    
    
    # 2 - import the timestamps of the events -----------------------------
    nameFile = 'TsEvents/'
    file_to_load_ev= filePath / mouse / Date / nameFile
    events = []
    # Iterate directory
    for file in os.listdir(file_to_load_ev):
        # check only text files
        if file.endswith('TS.csv'):
            events.append(file)
    print(events)
    
    
    # 3 - prepare the dataframe contining spikes -----------------------------
    
    # 3.0 Look for the list of single neurons and sort them
    namesub_0 = 'Sorting' 
    namesub_1 = 'all_sorters' 
    namesub_2 = '0' 
    if df['probe'][1] != 'tetrodes':
        namesub_3 = 'kilosort3/' 
    else:
        namesub_3 = 'kilosort2/' 
    file_to_load = filePath / mouse / Date / namesub_0 / namesub_1 / namesub_2 / namesub_3
    
    # list to store files
    neurons = []
    # Iterate directory
    for file in os.listdir(file_to_load):
        # check only text files
        if (file.startswith('SU') and file.endswith('.csv')):
            neurons.append(file)
    test_list = neurons
    del neurons
    test_list.sort(key=lambda test_string : list(
        map(int, re.findall(r'\d+', test_string)))[0])
    neurons = test_list
    del test_list
    print(neurons)
    
    # 3.1 iterate through each neuron -> events
    
    timepoint = np.arange(-2, 2+0.001, 0.001)
    AllN = pd.DataFrame()
    for n in tqdm(range(len(neurons))):
        neuron_to_load = file_to_load / neurons[n]
        spikes = np.ravel(np.array(pd.read_csv(neuron_to_load,index_col=0)))
        
        MainN = pd.DataFrame()  # we generate an empty dataframe that will contains the data of the  curr neuron   
        for e in range(len(events)):
            if e == 0:
                label = 0 # cold or Citric Acid
            if e == 1:
                label = 1 # room or NaCl
            if e == 2:
                label = 2 # hot oR Quinine
            if e == 3:
                label = 3 #  Sucrose
            if e == 4:
                label = 4 #  Water
                
            event_to_load = file_to_load_ev / events[e]
            ev = np.ravel(np.array(pd.read_csv(event_to_load)))
            # generate raster 
            unit={}
            unit[events[e]]={}
            for j in range(len(ev)):
                temp = []
                temp = list(np.ravel(spikes[np.where((spikes>=ev[j]-2)&(spikes<=ev[j]+2))]))
                unit[events[e]][j] = temp
            s = pd.Series(unit[events[e]])
        
            #
            fin_T = pd.DataFrame()
            fin   = list()
            for tp in range(len(timepoint)-1):
                if tp == 0:
                    Tr = list(range(0,len(ev)))
                    for trials in range(len(ev)):
                        temp_sp = spikes[np.where((spikes>=ev[trials]-2)&(spikes<=ev[trials]+2))]-ev[trials]
                        if np.any(np.logical_and(temp_sp>=timepoint[tp], temp_sp<=timepoint[tp+1])):
                            fin.append(1.0)
                        else:
                            fin.append(0.0)
                    fin_T['Recording Type'] = ['Neuron'] * len(ev)
                    fin_T['MouseID'] = [mouse] * len(ev)  
                    fin_T['Date'] = [Date] * len(ev)  
                    fin_T['n_ID'] = [neurons[n]] * len(ev) 
                    fin_T['Taste'] = [label] * len(ev)       
                    fin_T['Neuron'] = [n] * len(ev)  
                    fin_T['Trial'] = Tr
                    fin_T[int(round(timepoint[tp]*1000,4))] = fin
                    fin   = list()
            
                else:
                    for trials in range(len(ev)):
                        temp_sp = spikes[np.where((spikes>=ev[trials]-2)&(spikes<=ev[trials]+2))]-ev[trials]
                        if np.any(np.logical_and(temp_sp>=timepoint[tp], temp_sp<=timepoint[tp+1])):
                            fin.append(1.0)
                        else:
                            fin.append(0.0)
                    temp = pd.DataFrame()
                    temp[int(round(timepoint[tp]*1000,4))] = fin
                    fin_T = pd.concat((fin_T,temp),axis=1)
                    fin = list()
        
            if e == 0:
                MainN = fin_T
            else:
                MainN = pd.concat((MainN,fin_T),axis=0,ignore_index=True)
        if n == 0:
            AllN = MainN
        else:
            AllN = pd.concat((AllN,MainN),axis=0,ignore_index=True)
    
    nameFile_2 = 'SU_Analysis/AllN.pickle'
    file_to_load2= filePath / mouse / Date / nameFile_2
    with open(file_to_load2, 'wb') as handle:
        pickle.dump(AllN, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return AllN





    
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



