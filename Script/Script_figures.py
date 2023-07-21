#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:40:57 2023

@author: robertovincis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:37:18 2022

@author: robertovincis
"""
def Figure_1F(First_n_raster,First_n_psth_1,First_n_psth_2,First_n_psth_3,First_n_psth_4,First_n_psth_5, min_trialN_1,
              Second_n_raster,Second_n_psth_1,Second_n_psth_2,Second_n_psth_3,Second_n_psth_4,Second_n_psth_5, min_trialN_2,
):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    inner_1 = [['C'],['C1']]
    inner_2 = [['D'],['D1']]

    fig, axs = plt.subplot_mosaic([[inner_1,inner_2]],
                              constrained_layout=True,figsize=(7,5))
    colors = sns.color_palette("Paired")
    #colours = sns.color_palette('rocket')
    
    for label, ax in axs.items():
        if label == 'C':
            ax.eventplot(First_n_raster,color=[colors[5]]*min_trialN_1 + [colors[3]]*min_trialN_1 + [colors[1]]*min_trialN_1 + 
                                  [colors[7]]*min_trialN_1 + [colors[9]]*min_trialN_1)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.text(-0.1, 1.1, 'D', transform=ax.transAxes,size=12, weight='bold')
            ax.set_title('Example Neuron#1')

        if label == 'C1':
            ax.plot(First_n_psth_1.mean()*150,color = colors[5],label = 'W')
            ax.plot(First_n_psth_2.mean()*150,color = colors[3],label = 'N')
            ax.plot(First_n_psth_3.mean()*150,color = colors[1],label = 'C')
            ax.plot(First_n_psth_4.mean()*150,color = colors[7],label = 'Q')
            ax.plot(First_n_psth_5.mean()*150,color = colors[9],label = 'S')
            ax.set_xlim(-500, 2000)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'D':
            ax.eventplot(Second_n_raster,color=[colors[5]]*min_trialN_2 + [colors[3]]*min_trialN_2 + [colors[1]]*min_trialN_2 + 
                                  [colors[7]]*min_trialN_2 + [colors[9]]*min_trialN_2)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_title('Example Neuron#2')
            #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'D1':
            ax.plot(Second_n_psth_1.mean()*150,color = colors[5],label = 'W')
            ax.plot(Second_n_psth_2.mean()*150,color = colors[3],label = 'N')
            ax.plot(Second_n_psth_3.mean()*150,color = colors[1],label = 'C')
            ax.plot(Second_n_psth_4.mean()*150,color = colors[7],label = 'Q')
            ax.plot(Second_n_psth_5.mean()*150,color = colors[9],label = 'S')
            ax.set_xlim(-500, 2000)
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})

    figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_1F.pdf")
    plt.savefig(figureid,transparent=True)
            
def Figure_1G(Third_n_raster,Third_n_psth_1,Third_n_psth_2,Third_n_psth_3, min_trialN_3,
              Fourth_n_raster,Fourth_n_psth_1,Fourth_n_psth_2,Fourth_n_psth_3, min_trialN_4):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    inner_1 = [['E'],['E1']]
    inner_2 = [['F'],['F1']]

    fig, axs = plt.subplot_mosaic([[inner_1,inner_2]],
                              constrained_layout=True,figsize=(7,5))
    #colors = sns.color_palette("Paired")
    colours = sns.color_palette('rocket')
    
    for label, ax in axs.items():    
       if label == 'E':
           ax.eventplot(Third_n_raster,color=[colours[1]]*min_trialN_3 + [colours[3]]*min_trialN_3 + [colours[5]]*min_trialN_3)
           ax.set_xlim(-500, 2000)
           ax.spines['left'].set_visible(False)   
           ax.spines['bottom'].set_visible(False)
           ax.axes.yaxis.set_ticks([])
           ax.axes.xaxis.set_ticks([])
           ax.set_ylabel('Trials')
           ax.set_title('Example Neuron#3')
           ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
       if label == 'E1':
           #ax.plot(Third_n_psth_1.mean()*1000,color = colours[1],label = 'W-14')
           ax.plot(Third_n_psth_1.mean()*150,color = colours[1],label = 'W-14')
           #ax.plot(Third_n_psth_2.mean()*1000,color = colours[3],label = 'W-25')
           ax.plot(Third_n_psth_2.mean()*150,color = colours[3],label = 'W-25')
           #ax.plot(Third_n_psth_3.mean()*1000,color = colours[5],label = 'W-36')
           ax.plot(Third_n_psth_3.mean()*150,color = colours[5],label = 'W-36')
           ax.set_xlim(-500, 2000)
           ax.set_xlabel('Time (ms)')
           ax.set_ylabel('Firing rate (Hz)')
           ax.legend(prop={'size': 4})
       if label == 'F':
           ax.eventplot(Fourth_n_raster,color=[colours[1]]*min_trialN_3 + [colours[3]]*min_trialN_3 + [colours[5]]*min_trialN_3)
           ax.set_xlim(-500, 2000)
           ax.spines['left'].set_visible(False)   
           ax.spines['bottom'].set_visible(False)
           ax.axes.yaxis.set_ticks([])
           ax.axes.xaxis.set_ticks([])
           #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
           ax.set_title('Example Neuron#4')           
       if label == 'F1':
           ax.plot(Fourth_n_psth_1.mean()*150,color = colours[1],label = 'W-14')
           ax.plot(Fourth_n_psth_2.mean()*150,color = colours[3],label = 'W-25')
           ax.plot(Fourth_n_psth_3.mean()*150,color = colours[5],label = 'W-36')
           ax.set_xlim(-500, 2000)
           ax.set_xlabel('Time (ms)')
           #ax.set_ylabel('Firing rate (Hz)')
           ax.legend(prop={'size': 4})

    figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_1G.pdf")
    plt.savefig(figureid,transparent=True)
    
def Figure_2(Main_Temp,temp_active,temp_supp,temp_nonresp):
        
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import seaborn as sns
    import numpy as np
    
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    inner = [['D'],['D2'],['D3']]
    fig, axs = plt.subplot_mosaic([['A','B','C'],[inner,'E','E']],
                              constrained_layout=True,figsize=(6,5))
    colors = sns.color_palette("Paired")
    Main_Temp_NonResp = Main_Temp[(Main_Temp['resp_Cold_']==0) & (Main_Temp['resp_Hot_']==0) & (Main_Temp['resp_Room_']==0)] # filter the main dataframe into a new one containing only temp responsive neurons
    Main_Temp_Resp    = Main_Temp[(Main_Temp['resp_Cold_']==1) | (Main_Temp['resp_Hot_']==1) | (Main_Temp['resp_Room_']==1)]
    sizes = [100*len(Main_Temp_Resp),100*len(Main_Temp_NonResp)]
    explode = (0, 0.1,)
    labels = ['Res. neurons','Non res. neurons']
    for label, ax in axs.items():
        if label == 'A':
            ax.pie(sizes, explode = explode, colors=['black','gray'], labels=labels, autopct='%1.1f%%',shadow=False, startangle=90,textprops={'fontsize': 8, 'color':'w'})
            ax.axis('equal')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
            ax.text(-0.05, 0.95, 'Responsive neurons', transform=ax.transAxes, size=7, weight='bold', color = 'black')
            ax.text(-0.05, 0.90, 'Non responsive neurons', transform=ax.transAxes, size=7, weight='bold', color = 'gray')
            ax.set_title('All neurons recorded\n Temp. task (n=' + str(len(Main_Temp_Resp)+len(Main_Temp_NonResp)) + ')')
        if label == 'B': 
            x  = ['14˚C','25˚C','36˚C']
            y1 = [len(Main_Temp_Resp[Main_Temp_Resp['sign_Cold_']==1]) / len(Main_Temp),
                  len(Main_Temp_Resp[Main_Temp_Resp['sign_Room_']==1]) / len(Main_Temp),
                  len(Main_Temp_Resp[Main_Temp_Resp['sign_Hot_']==1]) / len(Main_Temp)]
            
            y2 = [len(Main_Temp_Resp[Main_Temp_Resp['sign_Cold_']==-1]) / len(Main_Temp),
                  len(Main_Temp_Resp[Main_Temp_Resp['sign_Room_']==-1]) / len(Main_Temp),
                  len(Main_Temp_Resp[Main_Temp_Resp['sign_Hot_']==-1]) / len(Main_Temp)]

            # plot bars in stack manner
            ax.bar(x, y1, color=colors[9])
            ax.bar(x, y2, bottom=y1, color = colors[3])
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
            ax.set_title('Fractions of responsive \n neurons')
            
        if label == 'C':
            y1_mean = np.mean(temp_active,axis=0)
            y1_err = np.std(temp_active,axis=0)/np.sqrt(len(temp_active))
            y2_mean = np.mean(temp_supp,axis=0)
            y2_err = np.std(temp_supp,axis=0)/np.sqrt(len(temp_supp))
            y3_mean = np.mean(temp_nonresp,axis=0)
            y3_err = np.std(temp_nonresp,axis=0)/np.sqrt(len(temp_nonresp))
            x = np.linspace(-1, 2, 26)
            ax.plot(x,y1_mean,color=colors[9])
            ax.fill_between(x, y1_mean+y1_err, y1_mean-y1_err,alpha=0.5,color=colors[9])
            ax.plot(x,y2_mean,color=colors[3])
            ax.fill_between(x, y2_mean+y2_err, y2_mean-y2_err,alpha=0.5,color=colors[3])
            ax.plot(x,y3_mean,color='gray')
            ax.fill_between(x, y3_mean+y3_err, y3_mean-y3_err,alpha=0.5,color='gray')
            ax.set_xlabel('Time (s)')
            ax.vlines(x=0, ymin=0.35, ymax=0.66, linestyle = 'dashed', colors = 'gray', linewidth=0.5)
            ax.set_ylim(0.35,0.66) 
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
            ax.set_title('Time course water \n responses')
            ax.set_ylabel('Firing Rate norm.')
            ax.set_xlabel('Time (s)')

           
        if label == 'E':
            filePath = '/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Data_for_analysis/'
            fileNameWav = 'waveforms_127.npy'
            wav = np.load(filePath + fileNameWav)
            wavMed = np.median(wav, axis=0)
            FtheMin = np.empty(shape=0)
            for w in range(wavMed.shape[1]):
                FtheMin = np.append(FtheMin, min(wavMed[:,w]))
                Wav2use = wavMed[:,np.argmin(FtheMin)]
                t2p_plot = np.argmax(Wav2use[np.argmin(Wav2use):]) + np.argmin(Wav2use)
            axins2 = inset_axes(ax, width=0.8, height=0.7,
                    bbox_to_anchor=(0.53, 0.5),
                    bbox_transform=ax.transAxes, loc=3, borderpad=0)
            axins2.plot(Wav2use, color = 'black',lw=0.8)
            axins2.set_xlim(np.argmin(Wav2use)-20,np.argmin(Wav2use)+45)
            axins2.hlines(y = 0, xmin=np.argmin(Wav2use), xmax=t2p_plot, linestyles = 'solid', color = 'red',lw=0.5)
            axins2.hlines(y = np.amin(Wav2use), xmin=140, xmax=150, linestyles = 'solid', color = 'black', lw=0.5)
            axins2.vlines(x = 150, ymin=np.amin(Wav2use), ymax=np.amin(Wav2use)+200, linestyles = 'solid', color = 'black', lw=0.5)
            axins2.text(0.85, 0.45, '0.1ms', transform=ax.transAxes, size=5.5, color = 'black')
            axins2.text(1.0, 0.52, "200\u03bcV", transform=ax.transAxes, size=5.5, color = 'black', rotation = "90")
            axins2.spines['bottom'].set_visible(False)
            axins2.spines['left'].set_visible(False)
            axins2.get_xaxis().set_visible(False)
            axins2.get_yaxis().set_visible(False)            
            ax.scatter(Main_Temp[Main_Temp['probe']=='tetrodes']['firing_rate'],Main_Temp[Main_Temp['probe']=='tetrodes']['Tr.2 Peak (ms)'], c = 'red', alpha=0.5, s=15,edgecolors='black')
            ax.scatter(Main_Temp[Main_Temp['probe']!='tetrodes']['firing_rate'],Main_Temp[Main_Temp['probe']!='tetrodes']['Tr.2 Peak (ms)'], c = 'gray', alpha=0.5, s=15,edgecolors='black')
            ax.set_ylabel('Trough to peak (ms)')
            ax.set_xlabel('Firing Rate (Hz)')
            ax.set_ylim(0,1.41)
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
            ax.text(0.05, 0.94, 'Tetrodes', transform=ax.transAxes, size=8, color = colors[5])
            ax.text(0.05, 0.87, 'Probes', transform=ax.transAxes, size=8, color = 'gray')
            ax.set_title('Waveform and firing \n properties')

        if label == 'D':
            sns.histplot(Main_Temp_Resp['latCold_'].dropna(), ax=ax)
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
            ax.vlines(x=Main_Temp_Resp['latCold_'].dropna().mean(), ymin = 0, ymax = 80, color=colors[1], linestyles='dashed', lw=0.5)
            ax.set_ylabel('Count')
            ax.set_title('Latency')
            ax.set_xlim(0,800)
            
        if label == 'D2':
            sns.histplot(Main_Temp_Resp['latRoom_'].dropna(), ax=ax)
            ax.vlines(x=Main_Temp_Resp['latRoom_'].dropna().mean(), ymin = 0, ymax = 80, color=colors[3], linestyles='dashed', lw=0.5)
            ax.set_ylabel('Count')
            ax.set_xlim(0,800)
            
        if label == 'D3':
            sns.histplot(Main_Temp_Resp['latHot_'].dropna(), ax=ax)
            ax.vlines(x=Main_Temp_Resp['latHot_'].dropna().mean(), ymin = 0, ymax = 80, color=colors[5], linestyles='dashed', lw=0.5)
            ax.set_ylabel('Count')
            ax.set_xlabel('Time (ms)')
            ax.set_xlim(0,800)

    figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_2.pdf")
    plt.savefig(figureid,transparent=True)
    
def Figure_3(Third_n_raster,Third_n_psth_1,Third_n_psth_2,Third_n_psth_3, min_trialN_3,
                                   Fourth_n_raster,Fourth_n_psth_1,Fourth_n_psth_2,Fourth_n_psth_3, min_trialN_4,
                                   Sel,nonSel,Tot_Resp,
                                   one_d,two_d,three_d,
                                   one_d2,two_d2,three_d2,
                                   df_test_1,df_test_11,df_test_111,
                                   df_test_2,df_test_22,df_test_222,
                                   df_test_3,df_test_4):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    inner_1 = [['A','B'],['A1','B1']]
    inner_3 = [['D','D1']]
    inner_2 = [['C',inner_3],['E','F']]
    
    colours = sns.color_palette('rocket')
    colours_1 = sns.color_palette('mako')
    colours_2 = sns.color_palette("icefire")
    
    fig, axs = plt.subplot_mosaic([[inner_1,inner_2]],
                              constrained_layout=True,figsize=(8,4))
    
    sizes = [Sel,nonSel]
    explode = (0,0.05)
    
    for label, ax in axs.items():
        if label == 'A':
            ax.eventplot(Third_n_raster,color=[colours[1]]*min_trialN_3 + [colours[3]]*min_trialN_3 + [colours[5]]*min_trialN_3)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Temperature independent')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'A1':
            ax.plot(Third_n_psth_1.mean()*150,color = colours[1],label = 'W-14')
            ax.plot(Third_n_psth_2.mean()*150,color = colours[3],label = 'W-25')
            ax.plot(Third_n_psth_3.mean()*150,color = colours[5],label = 'W-36')
            ax.set_xlim(-500, 2000)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'B':
            ax.eventplot(Fourth_n_raster,color=[colours[1]]*min_trialN_4 + [colours[3]]*min_trialN_4 + [colours[5]]*min_trialN_4)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_title('Temperature dependent')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'B1':
            ax.plot(Fourth_n_psth_1.mean()*150,color = colours[1],label = 'W-14')
            ax.plot(Fourth_n_psth_2.mean()*150,color = colours[3],label = 'W-25')
            ax.plot(Fourth_n_psth_3.mean()*150,color = colours[5],label = 'W-36')
            ax.set_xlim(-500, 2000)
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'C':
            ax.pie(sizes, explode = explode, colors=[colours_1[3],colours_1[1]], autopct='%1.1f%%',shadow=False, startangle=90,textprops={'fontsize': 8, 'color':'w'})
            ax.axis('equal')
            ax.text(-0.09, 0.95, 'Temp. dependent', transform=ax.transAxes, size=7, weight='bold', color = colours_1[0])
            ax.text(0.4, 0.95, 'Temp. independent.', transform=ax.transAxes, size=7, weight='bold', color = colours_1[1])
            ax.set_title('Water responsive\n neurons (n=' + str(Tot_Resp) + ')')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
        if label == 'D': 
            ax.bar([1,2,3],[one_d,two_d,three_d],color = colours_1[3])
            ax.set_xlabel('N. of stim.')
            ax.set_ylabel('% of neurons')
            ax.spines.bottom.set_visible(False)
            ax.xaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.text(1, 1, '1'),ax.text(2, 1, '2'),ax.text(3, 1, '3')
#            ax.text(1-0.1, one+2, str(len(Resp_sel_final[Resp_sel_final['tuning_1']==1]))+'/'+str(len(Resp_sel_final))),ax.text(2, 1, '2'),ax.text(3, 1, '3')
#            ax.text(2-0.1, two+2, str(len(Resp_sel_final[Resp_sel_final['tuning_1']==2]))+'/'+str(len(Resp_sel_final))),ax.text(2, 1, '2'),ax.text(3, 1, '3')
#            ax.text(3-0.1, three+2, str(len(Resp_sel_final[Resp_sel_final['tuning_1']==3]))+'/'+str(len(Resp_sel_final))),ax.text(2, 1, '2'),ax.text(3, 1, '3')
            ax.set_title('Tuning of temp.-selective neurons')
        if label == 'D1':
            ax.bar([1,2,3],[one_d2,two_d2,three_d2],color = [colours[1],colours[3],colours[5]])
            ax.set_xlabel('Stimulus I.D.')
            #ax.set_ylabel('% of neurons')
            ax.spines.bottom.set_visible(False)
            ax.xaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
        if label == 'E':
            ax.plot(df_test_3.T,color=colours_2[3])
            ax.plot(df_test_4.T,color=colours_2[3])
            ax.plot(df_test_1.T,color=colours_2[5],alpha=0.6)
            ax.plot(df_test_11.T,color=colours_2[5],alpha=0.6)
            ax.plot(df_test_111.T,color=colours_2[5],alpha=0.6)

            ax.plot(df_test_2.T,color=colours_2[0],alpha=0.6)
            ax.plot(df_test_22.T,color=colours_2[0],alpha=0.6)
            ax.plot(df_test_222.T,color=colours_2[0],alpha=0.6)

            ax.spines.bottom.set_visible(False)
            ax.set_xlabel('Stimulus I.D.')
            ax.set_ylabel('Evoked FR normalized')
            ax.set_ylim(0,1)
            x = [0,1,2]
            xi = ['14C','25C','36']
            ax.set_xticks(x)
            ax.set_xticklabels(xi,minor=False)
            ax.set_title('Tuning of temp.-selective neurons')
        if label == 'F':
            y1 = len(df_test_1)+len(df_test_11)+len(df_test_111)
            y2 = len(df_test_2)+len(df_test_22)+len(df_test_222)
            y3 = len(df_test_3)+len(df_test_4)
            ax.bar(1,100*(y1/Sel),color=colours_2[5])
            ax.bar(1,100*(y2/Sel),bottom =100*(y1/Sel),color=colours_2[0])
            ax.bar(2,100*(y3/Sel),color=colours_2[3])
            ax.set_ylabel('Fraction temp. dependent neurons')
            ax.set_xlabel('Mode')
            ax.text(0.95,2,'Monotonic', rotation=90,color='w')
            ax.text(1.95,2,'Non Mono.', rotation=90,color='w')

            ax.spines.bottom.set_visible(False)
            ax.xaxis.set_ticks([])

    #figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_3.pdf")
    #plt.savefig(figureid,transparent=True)
    
def Figure_4(filePath):
    """

    """ 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    fig, axs = plt.subplot_mosaic([['A','B']],
                              constrained_layout=True,figsize=(6,3))
    colors = sns.color_palette("Paired")
    mean_SVM_df         = pd.read_csv(filePath/'Data/slidwin_decoding.csv')
    mean_SVM_df_shuffle = pd.read_csv(filePath/'Data/shuffled_slidwin_decoding.csv')
    time = np.round(np.arange(0,80,1),decimals=1)
    for label, ax in axs.items():
#       if label == 'B':
#            ax.plot(evok_dec)
#            ax.plot(base_dec)
#            ax.set_xlim(0,50)
#            ax.set_xlabel('Population Size')
#            ax.set_ylabel('Accuracy (%)')
        if label == 'A':
            mean_SVM_df.iloc[0,3:].rolling(5).mean().plot(ax=ax)
            mean_SVM_df.iloc[1,3:].rolling(5).mean().plot(ax=ax)
            mean_SVM_df.iloc[2,3:].rolling(5).mean().plot(ax=ax)

            mean_SVM_df_shuffle.iloc[:,1:].quantile(0.0001).rolling(5).mean().plot(ax=ax)
            mean_SVM_df_shuffle.iloc[:,1:].quantile(0.999).rolling(5).mean().plot(ax=ax)

            #ax.vlines(x=40, ymin = 0.2, ymax=0.75)
            # for plot -
            #a = np.array(mean_SVM_df.iloc[0,3:].rolling(5).mean(), dtype = float)
            #b = np.array(mean_SVM_df.iloc[1,3:].rolling(5).mean(), dtype = float)
            #c = np.array(mean_SVM_df.iloc[2,3:].rolling(5).mean(), dtype = float)
            #time = np.round(np.arange(0,80,1),decimals=1)
            #aa = np.array(mean_SVM_df_shuffle.iloc[:,1:].quantile(0.01).rolling(5).min(), dtype = float)
            #bb = np.array(mean_SVM_df_shuffle.iloc[:,1:].quantile(0.99).rolling(5).max(), dtype = float)
            #ax.plot(time,a, color = 'w')
            #ax.fill_between(time,b,c,alpha=1,color = colors[9])
            #ax.fill_between(time,aa,bb,alpha=0.3,color = colors[11])
            #ax.vlines(x = 38, ymin = 0.2, ymax=0.7)
            #x = [0,38,76]
            #xi = ['-2','0','2']
            #ax.set_xticks(x)
            #ax.set_xticklabels(xi,minor=False)
            #ax.plot(time,mean_SVM_df.iloc[0,2:])
            #ax.fill_between(time,mean_SVM_df_shuffle.iloc[:,1:].quantile(0.001),mean_SVM_df_shuffle.iloc[:,1:].quantile(0.999), alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0.2,0.72)
            ax.text(-0.2, 1.15, label, transform=ax.transAxes,size=12, weight='bold')       
            ax.set_title('Population decoding timecourse')

        if label == 'B':
          evoked = np.load(filePath/'Data/decoding_pop_evoked.npy', mmap_mode=None)
          evoked_shuffle = np.load(filePath/'Data/decoding_pop_evoked_shuffle.npy', mmap_mode=None)
          ax.plot(evoked,color = colors[9], marker="o")
          ax.plot(evoked_shuffle, color = colors[11],marker="o")
          ax.set_xlabel('Population size (neurons)')
          ax.set_ylabel('Accuracy')
          ax.set_ylim(0.2,1)
          ax.set_title('Population decoding 1.5 s after stimulus')
          ax.text(-0.2, 1.15, label, transform=ax.transAxes,size=12, weight='bold') 
          
    #figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_4.pdf")
    #plt.savefig(figureid,transparent=True)
    
def Figure_5(Fifth_n_raster,Fifth_n_psth_1,Fifth_n_psth_2,Fifth_n_psth_3, min_trialN_5,
             Sixth_n_raster,Sixth_n_psth_1,Sixth_n_psth_2,Sixth_n_psth_3, min_trialN_6,
             Resp_sel,Resp_unsel, Main_Temp):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import r2_score


    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    inner_1 = [['A','AA'],['A1','AA1']]
    #inner_3 = [['C','C1']]
    inner_2 = [['B','C'],['D','E']] 
    
    colours = sns.color_palette('rocket')
    colours_1 = sns.color_palette('mako')
    colours_2 = sns.color_palette("icefire")
    
    fig, axs = plt.subplot_mosaic([[inner_1,inner_2]],
                              constrained_layout=True,figsize=(12,4))
    
    for label, ax in axs.items():
        if label == 'A':
            ax.eventplot(Fifth_n_raster,color=[colours[1]]*min_trialN_5 + [colours[3]]*min_trialN_5 + [colours[5]]*min_trialN_5)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Temperature dependent \n dorsal')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'A1':
            ax.plot(Fifth_n_psth_1.mean()*150,color = colours[1],label = 'W-14')
            ax.plot(Fifth_n_psth_2.mean()*150,color = colours[3],label = 'W-25')
            ax.plot(Fifth_n_psth_3.mean()*150,color = colours[5],label = 'W-36')
            ax.set_xlim(-500, 2000)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'AA':
            ax.eventplot(Sixth_n_raster,color=[colours[1]]*min_trialN_6 + [colours[3]]*min_trialN_6 + [colours[5]]*min_trialN_6)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_title('Temperature dependent \n ventral')
            #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'AA1':
            ax.plot(Sixth_n_psth_1.mean()*150,color = colours[1],label = 'W-14')
            ax.plot(Sixth_n_psth_2.mean()*150,color = colours[3],label = 'W-25')
            ax.plot(Sixth_n_psth_3.mean()*150,color = colours[5],label = 'W-36')
            ax.set_xlim(-500, 2000)
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})   
            
        # if label == 'B':
        #     x = ['600-800µm', '400-600µm', '200-400µm', '0-200µm']
            
        #     y = [(len(Resp_sel[Resp_sel['DV_exact']>599])+len(Resp_unsel[Resp_unsel['DV_exact']>599]))/len(Main_Temp[Main_Temp['DV_exact']>599]),
        #          (len(Resp_sel[Resp_sel['DV_exact']>399])+len(Resp_unsel[Resp_unsel['DV_exact']<599]))/(len(Main_Temp[Main_Temp['DV_exact']>399])+len(Main_Temp[Main_Temp['DV_exact']<599])),
        #          (len(Resp_sel[Resp_sel['DV_exact']>199])+len(Resp_unsel[Resp_unsel['DV_exact']<399]))/(len(Main_Temp[Main_Temp['DV_exact']>199])+len(Main_Temp[Main_Temp['DV_exact']<399])),
        #          (len(Resp_sel[Resp_sel['DV_exact']>0])+len(Resp_unsel[Resp_unsel['DV_exact']<199]))/(len(Main_Temp[Main_Temp['DV_exact']>0])+len(Main_Temp[Main_Temp['DV_exact']<199]))]

        #     ax.barh(x, y, align='center', color = 'k')
        #     #ax.barh(1, [0.2:0.7], align='center')
        #     ax.invert_yaxis()  # labels read top-to-bottom
        #     ax.set_xlim(0,1)
        #     ax.set_ylabel ('Position DV axes')
        #     ax.set_xlabel ('Water responsive neurons')
        #     ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
            
        if label == 'B':
            x = ['600-800µm', '400-600µm', '200-400µm', '0-200µm']
            y = [len(Resp_sel[Resp_sel['DV_exact']>599]) / (len(Resp_sel[Resp_sel['DV_exact']>600])+len(Resp_unsel[Resp_unsel['DV_exact']>600])),
                  len(Resp_sel[(Resp_sel['DV_exact']>399) & (Resp_sel['DV_exact']<599)]) / (len(Resp_sel[(Resp_sel['DV_exact']>399) & (Resp_sel['DV_exact']<599)]) + len(Resp_unsel[(Resp_unsel['DV_exact']>399) & (Resp_unsel['DV_exact']<599)])),
                  len(Resp_sel[(Resp_sel['DV_exact']>199) & (Resp_sel['DV_exact']<399)]) / (len(Resp_sel[(Resp_sel['DV_exact']>199) & (Resp_sel['DV_exact']<399)]) + len(Resp_unsel[(Resp_unsel['DV_exact']>199) & (Resp_unsel['DV_exact']<399)])),
                  len(Resp_sel[(Resp_sel['DV_exact']>0) & (Resp_sel['DV_exact']<199)]) / (len(Resp_sel[(Resp_sel['DV_exact']>0) & (Resp_sel['DV_exact']<199)]) + len(Resp_unsel[(Resp_unsel['DV_exact']>0) & (Resp_unsel['DV_exact']<199)]))]

            ax.barh(x, y, align='center')
            #ax.barh(1, [0.2:0.7], align='center')
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlim(0,1)
            ax.set_ylabel ('Position DV axes')
            ax.set_xlabel ('Water responsive neurons')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
            
        if label == 'C':
            Resp_sel_probes = Resp_sel[Resp_sel['probe']!='tetrodes'].reset_index()
            x = 100*Resp_sel_probes['Overall SVM Score']
            y = Resp_sel_probes['DV_exact']
            model = np.polyfit(x, y, 1)
            ax.plot(x,y,'o',color = colours_1[3])
            predict = np.poly1d(model)
            x_lin_reg = range(30, 80)
            y_lin_reg = predict(x_lin_reg)
            ax.scatter(x, y)
            ax.plot(x_lin_reg, y_lin_reg, c = 'r')
            r2_Sc = r2_score(y, predict(x))
            x = [30,40,50,60,70,80]
            xi = ['0.3','0.4','0.5','0.6','0.7','0.8']
            ax.set_xticks(x)
            ax.set_xticklabels(xi,minor=False)
            ax.set_title('Decoding Perf. over \n DV position')
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Neuron position (µm)')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')

        if label == 'D':
            ax.bar([1,2,3,4],[len(Resp_sel[(Resp_sel['position']=='0V')|(Resp_sel['position']=='0D')]) / (len(Resp_sel[(Resp_sel['position']=='0V')|(Resp_sel['position']=='0D')])+len(Resp_unsel[(Resp_unsel['position']=='0V')|(Resp_unsel['position']=='0D')])), 
                              len(Resp_sel[(Resp_sel['position']=='1V')|(Resp_sel['position']=='1D')]) / (len(Resp_sel[(Resp_sel['position']=='1V')|(Resp_sel['position']=='1D')])+len(Resp_unsel[(Resp_unsel['position']=='1V')|(Resp_unsel['position']=='1D')])), 
                              len(Resp_sel[(Resp_sel['position']=='2V')|(Resp_sel['position']=='2D')]) / (len(Resp_sel[(Resp_sel['position']=='2V')|(Resp_sel['position']=='2D')])+len(Resp_unsel[(Resp_unsel['position']=='2V')|(Resp_unsel['position']=='2D')])), 
                              len(Resp_sel[(Resp_sel['position']=='3V')|(Resp_sel['position']=='3D')]) / (len(Resp_sel[(Resp_sel['position']=='3V')|(Resp_sel['position']=='3D')]) + len(Resp_unsel[(Resp_unsel['position']=='3V')|(Resp_unsel['position']=='3D')]) )], color = colours_1[3])
            ax.set_xlabel('Pos. relative to bregma')
            ax.set_ylabel('Water responsive neurons')
            ax.set_title('Temp. specificity across \n AP axes')
            ax.set_ylim(0,1)
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
            ax.arrow(2.5, 9, -1, 0,head_width=0.5, head_length=0.5)
            ax.arrow(2.5, 9, 1, 0,head_width=0.5, head_length=0.5)
            x = [1,2,3,4]
            xi = ['+1.2','+0.95','+0.7', '+0.45']
            ax.set_xticks(x)
            ax.set_xticklabels(xi,minor=False)
            
        if label == 'E':
            X_1 = np.random.uniform(0.9, 1.1, len(Resp_sel[(Resp_sel['position']=='0V')|(Resp_sel['position']=='0D')]['Overall SVM Score']))
            X_2 = np.random.uniform(1.9, 2.1, len(Resp_sel[(Resp_sel['position']=='1V')|(Resp_sel['position']=='1D')]['Overall SVM Score']))
            X_3 = np.random.uniform(2.9, 3.1, len(Resp_sel[(Resp_sel['position']=='2V')|(Resp_sel['position']=='2D')]['Overall SVM Score']))
            X_4 = np.random.uniform(3.9, 4.1, len(Resp_sel[(Resp_sel['position']=='3V')|(Resp_sel['position']=='3D')]['Overall SVM Score']))
            ax.plot(X_1,Resp_sel[(Resp_sel['position']=='0V')|(Resp_sel['position']=='0D')]['Overall SVM Score'],'o',color = colours_1[3])
            ax.plot(X_2,Resp_sel[(Resp_sel['position']=='1V')|(Resp_sel['position']=='1D')]['Overall SVM Score'],'o',color = colours_1[3])
            ax.plot(X_3,Resp_sel[(Resp_sel['position']=='2V')|(Resp_sel['position']=='2D')]['Overall SVM Score'],'o',color = colours_1[3])
            ax.plot(X_4,Resp_sel[(Resp_sel['position']=='3V')|(Resp_sel['position']=='3D')]['Overall SVM Score'],'o',color = colours_1[3])
            ax.set_xlim(0.5,4.5)
            x = [1,2,3,4]
            xi = ['+1.2','+0.95','+0.7', '+0.45']
            ax.set_xticks(x)
            ax.set_xticklabels(xi,minor=False)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Pos. relative to bregma')
            ax.set_title('Decoding Perf. over \n AP position')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')

    #figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_5.pdf")
    #plt.savefig(figureid,transparent=True)
    #return (r2_Sc)

def Figure_6(Fifth_n_raster, Fifth_n_psth_1, Fifth_n_psth_2, Fifth_n_psth_3, Fifth_n_psth_4, min_trialN_5,
             MWS,
             CW,CS,HW,HS):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import r2_score


    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    # inner_1 = [['A','B','B','D','E'],['A1','B1','C1','D1','E1']]

    
    # colours = sns.color_palette('rocket')
    # colours_1 = sns.color_palette('mako')
    # colours_2 = sns.color_palette("icefire")
    
    # fig, axs = plt.subplot_mosaic([[inner_1]],
    #                           constrained_layout=True,figsize=(12,4))
    
    inner_3 = [['BB'],['BB1']]
    inner_1 = [['A',inner_3]]
    inner_2 = [['C','C1','D']] 
    
    # colours = sns.color_palette('rocket')
    # colours_1 = sns.color_palette('mako')
    # colours_2 = sns.color_palette("icefire")
    
    fig, axs = plt.subplot_mosaic([[inner_1],[inner_2]],
                              constrained_layout=True,figsize=(5,5))
    
    for label, ax in axs.items():
        if label == 'BB':
            #ax.eventplot(Fifth_n_raster,color=[colours[0]]*min_trialN_5 + [colours[1]]*min_trialN_5 + [colours[2]]*min_trialN_5 + [colours[3]]*min_trialN_5)
            ax.eventplot(Fifth_n_raster,color=['blue']*min_trialN_5 + ['red']*min_trialN_5 + ['navy']*min_trialN_5 + ['firebrick']*min_trialN_5)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.text(-0.1, 1.1, 'B', transform=ax.transAxes,size=12, weight='bold')
        if label == 'BB1':
            #ax.plot(Fifth_n_psth_1.mean()*1000,color = colours[0],label = 'Saliva-14C˚')
            ax.plot(Fifth_n_psth_1.mean()*150,color = 'blue',label = 'Saliva-14C˚')
            #ax.plot(Fifth_n_psth_2.mean()*1000,color = colours[1],label = 'Saliva-36C˚')
            ax.plot(Fifth_n_psth_2.mean()*150,color = 'red',label = 'Saliva-36C˚')
            #ax.plot(Fifth_n_psth_3.mean()*1000,color = colours[2],label = 'Water-14C˚')
            ax.plot(Fifth_n_psth_3.mean()*150,color = 'navy',label = 'Water-14C˚')
            #ax.plot(Fifth_n_psth_4.mean()*1000,color = colours[3],label = 'Water-36C˚')
            ax.plot(Fifth_n_psth_4.mean()*150,color = 'firebrick',label = 'Water-36C˚')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 12)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        # if label == 'CC':
        #     ax.eventplot(Sixth_n_raster,color=['blue']*min_trialN_6 + ['red']*min_trialN_6 + ['navy']*min_trialN_6 + ['firebrick']*min_trialN_6)
        #     ax.set_xlim(-500, 2000)
        #     ax.spines['left'].set_visible(False)   
        #     ax.spines['bottom'].set_visible(False)
        #     ax.axes.yaxis.set_ticks([])
        #     ax.axes.xaxis.set_ticks([])
        #     #ax.set_ylabel('Trials')
        #     #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        # if label == 'CC1':
        #     ax.plot(Sixth_n_psth_1.mean()*150,color = 'blue',label = 'Saliva-14C˚')
        #     ax.plot(Sixth_n_psth_2.mean()*150,color = 'red',label = 'Saliva-36C˚')
        #     ax.plot(Sixth_n_psth_3.mean()*150,color = 'navy',label = 'Water-14C˚')
        #     ax.plot(Sixth_n_psth_4.mean()*150,color = 'firebrick',label = 'Water-36C˚')
        #     ax.set_xlim(-500, 1500)
        #     ax.set_ylim(0, 30)
        #     ax.set_xlabel('Time (ms)')
        #     #ax.set_ylabel('Firing rate (Hz)')
        #     #ax.legend(prop={'size': 4})
        if label == 'D':
            x = 100*CS[:,6:11].mean(axis=1)
            y = 100*CW[:,6:11].mean(axis=1)
            
            model = np.polyfit(x, y, 1)
            predict = np.poly1d(model)
            x_lin_reg = range(1, 100)
            y_lin_reg = predict(x_lin_reg)
            plt.scatter(x, y, color= 'blue', alpha=0.5)
            plt.plot(x_lin_reg, y_lin_reg, c = 'blue')
            plt.xlabel('Evoked firing rate (norm.) - AS')
            plt.ylabel('Evoked firing rate (norm.) - W')
            r2_Sc_c = r2_score(y, predict(x))
            
            x = 100*HS[:,6:11].mean(axis=1)
            y = 100*HW[:,6:11].mean(axis=1)
            
            model = np.polyfit(x, y, 1)
            predict = np.poly1d(model)
            x_lin_reg = range(1, 100)
            y_lin_reg = predict(x_lin_reg)
            plt.scatter(x, y, color= 'red', alpha=0.5)
            plt.plot(x_lin_reg, y_lin_reg, c = 'red')
            plt.xlabel('Evoked firing rate (norm.) - AS')
            plt.ylabel('Evoked firing rate (norm.) - W')
            r2_Sc_h = r2_score(y, predict(x))
            
        if label == 'C':
            time = np.round(np.arange(-0.5,2,0.25),decimals=2)
            ax.fill_between(time, CW[np.where(CW[:,5:].mean(axis=1)<0.5)].mean(axis=0), CW[np.where(CW[:,5:].mean(axis=1)<0.5)].mean(axis=0)+CW[np.where(CW[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(CW)), color = 'navy',alpha=0.2)
            ax.fill_between(time, CW[np.where(CW[:,5:].mean(axis=1)<0.5)].mean(axis=0), CW[np.where(CW[:,5:].mean(axis=1)<0.5)].mean(axis=0)-CW[np.where(CW[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(CW)),color = 'navy',alpha=0.2)
            ax.plot(time, CW[np.where(CW[:,5:].mean(axis=1)<0.5)].mean(axis=0), color= 'navy')
            
            ax.fill_between(time, CS[np.where(CS[:,5:].mean(axis=1)<0.5)].mean(axis=0), CS[np.where(CS[:,5:].mean(axis=1)<0.5)].mean(axis=0)+CS[np.where(CS[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(CS)), color = 'blue',alpha=0.2)
            ax.fill_between(time, CS[np.where(CS[:,5:].mean(axis=1)<0.5)].mean(axis=0), CS[np.where(CS[:,5:].mean(axis=1)<0.5)].mean(axis=0)-CS[np.where(CS[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(CS)),color = 'blue',alpha=0.2)
            ax.plot(time, CS[np.where(CS[:,5:].mean(axis=1)<0.5)].mean(axis=0), color= 'blue')
            
            ax.fill_between(time, CW[np.where(CW[:,5:].mean(axis=1)>0.5)].mean(axis=0), CW[np.where(CW[:,5:].mean(axis=1)>0.5)].mean(axis=0)+CW[np.where(CW[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(CW)), color = 'navy',alpha=0.2)
            ax.fill_between(time, CW[np.where(CW[:,5:].mean(axis=1)>0.5)].mean(axis=0), CW[np.where(CW[:,5:].mean(axis=1)>0.5)].mean(axis=0)-CW[np.where(CW[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(CW)),color = 'navy',alpha=0.2)
            ax.plot(time, CW[np.where(CW[:,5:].mean(axis=1)>0.5)].mean(axis=0), color= 'navy')
            
            ax.fill_between(time, CS[np.where(CS[:,5:].mean(axis=1)>0.5)].mean(axis=0), CS[np.where(CS[:,5:].mean(axis=1)>0.5)].mean(axis=0)+CS[np.where(CS[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(CS)), color = 'blue',alpha=0.2)
            ax.fill_between(time, CS[np.where(CS[:,5:].mean(axis=1)>0.5)].mean(axis=0), CS[np.where(CS[:,5:].mean(axis=1)>0.5)].mean(axis=0)-CS[np.where(CS[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(CS)),color = 'blue',alpha=0.2)
            ax.plot(time, CS[np.where(CS[:,5:].mean(axis=1)>0.5)].mean(axis=0), color= 'blue')
            
            ax.vlines(x = 0, ymin = 0.2, ymax = 0.8, linestyle = 'dashed', colors = 'gray')
            ax.hlines(xmin = -0.5, y = 0.5, xmax = 1.7, linestyle = 'dashed', colors = 'gray')
         
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing rate norm.')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
            ax.set_ylim(0.2,0.8)
           
        if label == 'C1':
            time = np.round(np.arange(-0.5,2,0.25),decimals=2)
            ax.fill_between(time, HW[np.where(HW[:,5:].mean(axis=1)<0.5)].mean(axis=0), HW[np.where(HW[:,5:].mean(axis=1)<0.5)].mean(axis=0)+HW[np.where(HW[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(HW)), color = 'firebrick',alpha=0.2)
            ax.fill_between(time, HW[np.where(HW[:,5:].mean(axis=1)<0.5)].mean(axis=0), HW[np.where(HW[:,5:].mean(axis=1)<0.5)].mean(axis=0)-HW[np.where(HW[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(HW)),color = 'firebrick',alpha=0.2)
            ax.plot(time, HW[np.where(HW[:,5:].mean(axis=1)<0.5)].mean(axis=0), color= 'firebrick')
            
            ax.fill_between(time, HS[np.where(HS[:,5:].mean(axis=1)<0.5)].mean(axis=0), HS[np.where(HS[:,5:].mean(axis=1)<0.5)].mean(axis=0)+HS[np.where(HS[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(HS)), color = 'r',alpha=0.2)
            ax.fill_between(time, HS[np.where(HS[:,5:].mean(axis=1)<0.5)].mean(axis=0), HS[np.where(HS[:,5:].mean(axis=1)<0.5)].mean(axis=0)-HS[np.where(HS[:,5:].mean(axis=1)<0.5)].std(axis=0)/np.sqrt(len(HS)),color = 'r',alpha=0.2)
            ax.plot(time, HS[np.where(HS[:,5:].mean(axis=1)<0.5)].mean(axis=0), color= 'r')
            
            ax.fill_between(time, HW[np.where(HW[:,5:].mean(axis=1)>0.5)].mean(axis=0), HW[np.where(HW[:,5:].mean(axis=1)>0.5)].mean(axis=0)+HW[np.where(HW[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(HW)), color = 'firebrick',alpha=0.2)
            ax.fill_between(time, HW[np.where(HW[:,5:].mean(axis=1)>0.5)].mean(axis=0), HW[np.where(HW[:,5:].mean(axis=1)>0.5)].mean(axis=0)-HW[np.where(HW[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(HW)),color = 'firebrick',alpha=0.2)
            ax.plot(time, HW[np.where(HW[:,5:].mean(axis=1)>0.5)].mean(axis=0), color= 'firebrick')
            
            ax.fill_between(time, HS[np.where(HS[:,5:].mean(axis=1)>0.5)].mean(axis=0), HS[np.where(HS[:,5:].mean(axis=1)>0.5)].mean(axis=0)+HS[np.where(HS[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(HS)), color = 'r',alpha=0.2)
            ax.fill_between(time, HS[np.where(HS[:,5:].mean(axis=1)>0.5)].mean(axis=0), HS[np.where(HS[:,5:].mean(axis=1)>0.5)].mean(axis=0)-HS[np.where(HS[:,5:].mean(axis=1)>0.5)].std(axis=0)/np.sqrt(len(HS)),color = 'r',alpha=0.2)
            ax.plot(time, HS[np.where(HS[:,5:].mean(axis=1)>0.5)].mean(axis=0), color= 'r')
            
            ax.vlines(x = 0, ymin = 0.2, ymax = 0.8, linestyle = 'dashed', colors = 'gray')
            ax.hlines(xmin = -0.5, y = 0.5, xmax = 1.7, linestyle = 'dashed', colors = 'gray')
           
            ax.set_ylim(0.2,0.8)
            ax.set_xlabel('Time (s)')

    #figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_6.pdf")
    #plt.savefig(figureid,transparent=True) 
    #return(r2_Sc_c,r2_Sc_h)  

def Figure_7(Fifth_n_raster,Fifth_n_psth_1,Fifth_n_psth_2,Fifth_n_psth_3, Fifth_n_psth_4, min_trialN_5,
             Sixth_n_raster, Sixth_n_psth_1, Sixth_n_psth_2,min_trialN_6,
             Seventh_n_raster,Seventh_n_psth_1,Seventh_n_psth_2,Seventh_n_psth_3, Seventh_n_psth_4, min_trialN_7,
             Eight_n_raster, Eight_n_psth_1, Eight_n_psth_2,min_trialN_8,
             Nineth_n_raster,Nineth_n_psth_1,Nineth_n_psth_2,Nineth_n_psth_3, Nineth_n_psth_4, min_trialN_9,
             Ten_n_raster, Ten_n_psth_1, Ten_n_psth_2,min_trialN_10,
             Eleven_n_raster,Eleven_n_psth_1,Eleven_n_psth_2,Eleven_n_psth_3, Eleven_n_psth_4, min_trialN_11,
             Twelve_n_raster, Twelve_n_psth_1, Twelve_n_psth_2,min_trialN_12,
             MTT):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import r2_score


    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    
    inner_1 = [['A'],['AA']]
    inner_2 = [['B'],['BB']]
    inner_3 = [['C'],['CC']]
    inner_4 = [['D'],['DD']]
    inner_5 = [['E'],['EE']]
    inner_6 = [['F'],['FF']]
    inner_7 = [['G'],['GG']]
    inner_8 = [['H'],['HH']]
   
    colours = sns.color_palette('rocket')
    colors = sns.color_palette('Paired')
    
    fig, axs = plt.subplot_mosaic([['Z','Y','YY','X'],[inner_1,inner_2,inner_3,inner_4],[inner_5,inner_6,inner_7,inner_8]],
                              constrained_layout=True,figsize=(7,7))
    
    Taste_and_temp = len(MTT[(MTT['flag_Taste']==True) & (MTT['flag_Temp']==True)])
    Taste_only = len(MTT[MTT['flag_Taste']==True]) - Taste_and_temp

    sizes = [100*Taste_and_temp,100*Taste_only]
    explode = (0, 0.1,)
    labels = ['Taste and Temp.','Taste only']

    for label, ax in axs.items():
        if label == 'X':
            ax.pie(sizes, explode = explode, colors=['black','gray'], autopct='%1.1f%%',shadow=False, startangle=90,textprops={'fontsize': 8, 'color':'w'})
            ax.axis('equal')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=12, weight='bold')
            ax.text(-0.05, 0.95, 'Taste and Temp.', transform=ax.transAxes, size=7, weight='bold', color = 'black')
            ax.text(-0.05, 0.90, 'Taste only', transform=ax.transAxes, size=7, weight='bold', color = 'gray')
            ax.set_title('All taste-selective neurons (n=' + str(Taste_and_temp+Taste_only) + ')')
        if label == 'A':
            ax.eventplot(Fifth_n_raster,color=[colors[1]]*min_trialN_5 + [colors[3]]*min_trialN_5 + [colors[7]]*min_trialN_5 + [colors[9]]*min_trialN_5,
                         linewidths=1)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Neuron#1')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'AA':
            ax.plot(Fifth_n_psth_1.mean()*150,color = colors[1],label = 'CitricAcid')
            ax.plot(Fifth_n_psth_2.mean()*150,color = colors[3],label = 'NaCl')
            ax.plot(Fifth_n_psth_3.mean()*150,color = colors[7],label = 'Quinine')
            ax.plot(Fifth_n_psth_4.mean()*150,color = colors[9],label = 'Sucrose')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 13)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'B':
            ax.eventplot(Sixth_n_raster,color=[colours[1]]*min_trialN_6 + [colours[5]]*min_trialN_6)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            #ax.set_ylabel('Trials')
            ax.set_title('Neuron#1')
            #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'BB':
            ax.plot(Sixth_n_psth_1.mean()*150,color = colours[1],label = 'Water 14˚C')
            ax.plot(Sixth_n_psth_2.mean()*150,color = colours[5],label = 'Water 36˚C')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 13)
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'C':
            ax.eventplot(Seventh_n_raster,color=[colors[1]]*min_trialN_7 + [colors[3]]*min_trialN_7 + [colors[7]]*min_trialN_7 + [colors[9]]*min_trialN_7,
                          linewidths=0.8)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Neuron#2')
            ax.text(-0.1, 1.1, 'B', transform=ax.transAxes,size=12, weight='bold')
        if label == 'CC':
            ax.plot(Seventh_n_psth_1.mean()*150,color = colors[1],label = 'CitricAcid')
            ax.plot(Seventh_n_psth_2.mean()*150,color = colors[3],label = 'NaCl')
            ax.plot(Seventh_n_psth_3.mean()*150,color = colors[7],label = 'Quinine')
            ax.plot(Seventh_n_psth_4.mean()*150,color = colors[9],label = 'Sucrose')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 8)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'D':
            ax.eventplot(Eight_n_raster,color=[colours[1]]*10 + [colours[5]]*11,
                          linewidths=0.8)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            #ax.set_ylabel('Trials')
            ax.set_title('Neuron#2')
            #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'DD':
            ax.plot(Eight_n_psth_1.mean()*150,color = colours[1],label = 'Cold')
            ax.plot(Eight_n_psth_2.mean()*150,color = colours[5],label = 'Hot')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 8)
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'E':
            ax.eventplot(Nineth_n_raster,color=[colors[1]]*min_trialN_9 + [colors[3]]*min_trialN_9 + [colors[7]]*min_trialN_9 + [colors[9]]*min_trialN_9,
                          linewidths=1)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Neuron#3')
            ax.text(-0.1, 1.1, 'C', transform=ax.transAxes,size=12, weight='bold')
        if label == 'EE':
            ax.plot(Nineth_n_psth_1.mean()*150,color = colors[1],label = 'CitricAcid')
            ax.plot(Nineth_n_psth_2.mean()*150,color = colors[3],label = 'NaCl')
            ax.plot(Nineth_n_psth_3.mean()*150,color = colors[7],label = 'Quinine')
            ax.plot(Nineth_n_psth_4.mean()*150,color = colors[9],label = 'Sucrose')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 6)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'F':
            ax.eventplot(Ten_n_raster,color=[colours[1]]*min_trialN_10 + [colours[5]]*min_trialN_10)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            #ax.set_ylabel('Trials')
            ax.set_title('Neuron#3')
            #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'FF':
            ax.plot(Ten_n_psth_1.mean()*150,color = colours[1],label = 'Cold')
            ax.plot(Ten_n_psth_2.mean()*150,color = colours[5],label = 'Hot')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 6)
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'G':
            ax.eventplot(Eleven_n_raster,color=[colors[1]]*min_trialN_11 + [colors[3]]*min_trialN_11 + [colors[7]]*min_trialN_11 + [colors[9]]*min_trialN_11,
                          linewidths=1)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Neuron#4')
            ax.text(-0.1, 1.1, 'D', transform=ax.transAxes,size=12, weight='bold')
        if label == 'GG':
            ax.plot(Eleven_n_psth_1.mean()*150,color = colors[1],label = 'CitricAcid')
            ax.plot(Eleven_n_psth_2.mean()*150,color = colors[3],label = 'NaCl')
            ax.plot(Eleven_n_psth_3.mean()*150,color = colors[7],label = 'Quinine')
            ax.plot(Eleven_n_psth_4.mean()*150,color = colors[9],label = 'Sucrose')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 6)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})
        if label == 'H':
            ax.eventplot(Twelve_n_raster,color=[colours[1]]*min_trialN_12 + [colours[5]]*min_trialN_12)
            ax.set_xlim(-500, 2000)
            ax.spines['left'].set_visible(False)   
            ax.spines['bottom'].set_visible(False)
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticks([])
            ax.set_ylabel('Trials')
            ax.set_title('Neuron#4')
            #ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'HH':
            ax.plot(Twelve_n_psth_1.mean()*150,color = colours[1],label = 'Cold')
            ax.plot(Twelve_n_psth_2.mean()*150,color = colours[5],label = 'Hot')
            ax.set_xlim(-500, 2000)
            ax.set_ylim(0, 6)            
            ax.set_xlabel('Time (ms)')
            #ax.set_ylabel('Firing rate (Hz)')
            ax.legend(prop={'size': 4})

    figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_7.pdf")
    plt.savefig(figureid,transparent=True)  
    
def Figure_8(MTT):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import r2_score


    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"]= 300
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
     
    colors = ['goldenrod', 'seagreen']
    
    fig, axs = plt.subplot_mosaic([['A','B','B1'],['C','D','D1']],
                              constrained_layout=True,figsize=(5,4))
    df_t = MTT[MTT['flag_Taste']==True]

    for label, ax in axs.items():
        if label == 'A':
            ax.plot(df_t[df_t['flag_Temp']==True][['resp_CitricAcid','resp_NaCl','resp_Quinine','resp_Sucrose']].mean()*100,color=colors[0])
            ax.plot(df_t[(df_t['flag_Temp']==False)|(df_t['flag_Temp'].isnull())][['resp_CitricAcid','resp_NaCl','resp_Quinine','resp_Sucrose']].mean()*100, color=colors[1])
            ax.set_ylabel('Taste-selective neurons (%)')
            ax.set_title('Individual taste \n responsivness')
            ax.set_ylim(40,100)
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'B':
            x = df_t[df_t['flag_Temp']==True]['SI'].dropna()
            x_1 = df_t[(df_t['flag_Temp']==False)|(df_t['flag_Temp'].isnull())]['SI'].dropna()
            X_1 = np.random.uniform(0.8, 1.2, len(x))
            X_2 = np.random.uniform(1.8, 2.2, len(x_1))
            ax.plot(X_1,x,'o',color = colors[0],alpha=0.4)
            ax.plot(X_2,x_1,'o',color = colors[1],alpha=0.4)
            ax.boxplot([x,x_1])
            box = ax.boxplot([x,x_1],labels = ['Taste and temp.','Only Taste.'],
                             patch_artist=True,showfliers=False)
            # fill with colors            
            for b, c in zip(box['boxes'], colors):
                #b.set_alpha(0.6)
                b.set_edgecolor(c) # or try 'black'
                b.set_facecolor(c)
                b.set_linewidth(1)
                b.set_alpha(0.5)
                #b.whiskerprops(c)
            ax.set_ylabel('Sharpness index')
            ax.set_title('Breath of tuning')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'B1':
            k = len(x[(x>0)&(x<0.35)])/len(x)
            j = len(x_1[(x_1>0)&(x_1<0.35)])/len(x_1)
            kk = len(x[(x>0.65)&(x<1)])/len(x)
            jj = len(x_1[(x_1>0.65)&(x_1<1)])/len(x_1)
            ax.plot(1,k, 'o',color = colors[0])
            ax.plot(2,j,'o',color = colors[1])
            ax.plot(5,kk, 'o',color = colors[0])
            ax.plot(6,jj,'o',color = colors[1])
            ax.set_xlim(0,7)
            ax.set_ylim(0,0.55)
            ax.set_ylabel('Neurons (%)')
            ax.set_title('Broadly tuned taste neurons')
        if label == 'C':
            x = df_t[df_t['flag_Temp']==True]['PI'].dropna()
            x_1 = df_t[(df_t['flag_Temp']==False)|(df_t['flag_Temp'].isnull())]['PI'].dropna()
            X_1 = np.random.uniform(0.8, 1.2, len(x))
            X_2 = np.random.uniform(1.8, 2.2, len(x_1))
            ax.plot(X_1,x,'o',color = colors[0],alpha=0.4)
            ax.plot(X_2,x_1,'o',color = colors[1],alpha=0.4)
            ax.boxplot([x,x_1])
            box = ax.boxplot([x,x_1],labels = ['Taste and temp.','Only Taste.'],
                             patch_artist=True,showfliers=False)
            # fill with colors            
            for b, c in zip(box['boxes'], colors):
                #b.set_alpha(0.6)
                b.set_edgecolor(c) # or try 'black'
                b.set_facecolor(c)
                b.set_linewidth(1)
                b.set_alpha(0.5)
                #b.whiskerprops(c)
            ax.set_ylabel('Palatability index')
            ax.set_title('Palatability-related activity')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'D':
            x = df_t[df_t['flag_Temp']==True]['SVM_Taste'].dropna()
            x_1 = df_t[(df_t['flag_Temp']==False)|(df_t['flag_Temp'].isnull())]['SVM_Taste'].dropna()
            X_1 = np.random.uniform(0.8, 1.2, len(x))
            X_2 = np.random.uniform(1.8, 2.2, len(x_1))
            ax.plot(X_1,x,'o',color = colors[0],alpha=0.4)
            ax.plot(X_2,x_1,'o',color = colors[1],alpha=0.4)
            ax.boxplot([x,x_1])
            box = ax.boxplot([x,x_1],labels = ['Taste and temp.','Only Taste.'],
                             patch_artist=True,showfliers=False)
            # fill with colors            
            for b, c in zip(box['boxes'], colors):
                #b.set_alpha(0.6)
                b.set_edgecolor(c) # or try 'black'
                b.set_facecolor(c)
                b.set_linewidth(1)
                b.set_alpha(0.5)
                #b.whiskerprops(c)
            ax.set_ylabel('Taste decoding accuracy')
            ax.set_title('Amount of taste information')
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,size=12, weight='bold')
        if label == 'D1':
            k = len(x[(x>0.55)&(x<1)])/len(x)
            j = len(x_1[(x_1>0.55)&(x_1<1)])/len(x_1)
            #kk = len(x[(x>0.25)&(x<0.6)])/len(x)
            #jj = len(x_1[(x_1>0.25)&(x_1<0.6)])/len(x_1)
            ax.plot(1,k, 'o',color = colors[0])
            ax.plot(2,j,'o',color = colors[1])
            #ax.plot(5,kk, 'o',color = colors[0])
            #ax.plot(6,jj,'o',color = colors[1])
            ax.set_xlim(0,3)
            ax.set_ylim(0,0.5)
            ax.set_ylabel('Neurons (%)')
            ax.set_title('Broadly tuned taste neurons')
            
    figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_8.pdf")
    plt.savefig(figureid,transparent=True)  
            
def snippet_fig2_1(df_allN,df_allN_smooth,neuron=0, class_labels = ['Water','NaCl','CitricAcid','Quinine','Sucrose']):
    
    from Script.Script import get_spike_train_all,get_min_trial_numb
    import pandas as pd
    
    curr_n_ras = get_spike_train_all(df_allN, neuron)
    curr_n = get_spike_train_all(df_allN_smooth, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1

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

    if len(class_labels)==5:
        s_0 = pd.Series(unit_all[0])
        s_1 = pd.Series(unit_all[1])
        s_2 = pd.Series(unit_all[2])
        s_3 = pd.Series(unit_all[3])
        s_4 = pd.Series(unit_all[4])
        First_n = pd.Series(pd.concat((s_0,s_1,s_2,s_3,s_4),axis=0))
    
    min_index = df_allN_smooth.columns.get_loc('Trial') + 1
    copy_data = df_allN_smooth.copy()
    copy_data_N = copy_data[copy_data['Neuron']==neuron]

    if len(class_labels)==5:

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
        
    return First_n,copy_data_N_0_C,copy_data_N_1_C,copy_data_N_2_C,copy_data_N_3_C,copy_data_N_4_C, min_trialN

def snippet_fig2_2(df_allN,df_allN_smooth,neuron=0):
    
    from Script.Script import get_spike_train_all,get_min_trial_numb
    import pandas as pd
    
    curr_n_ras = get_spike_train_all(df_allN, neuron)
    curr_n = get_spike_train_all(df_allN_smooth, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1

    class_labels = ['Cold','Hot','Room']
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
        s_0 = pd.Series(unit_all[0]) # Cold
        s_1 = pd.Series(unit_all[2]) # Room
        s_2 = pd.Series(unit_all[1]) # Hot
        First_n = pd.Series(pd.concat((s_0,s_1,s_2),axis=0))
    
    min_index = df_allN_smooth.columns.get_loc('Trial') + 1
    copy_data = df_allN_smooth.copy()
    copy_data_N = copy_data[copy_data['Neuron']==neuron]

    if len(class_labels)==3:

        copy_data_N_0 = copy_data_N[copy_data_N['Taste']==0]
        copy_data_N_1 = copy_data_N[copy_data_N['Taste']==2]
        copy_data_N_2 = copy_data_N[copy_data_N['Taste']==1]
      
        
        copy_data_N_0_C = copy_data_N_0.copy() 
        copy_data_N_0_C.drop(copy_data_N_0_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_1_C = copy_data_N_1.copy() 
        copy_data_N_1_C.drop(copy_data_N_1_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_2_C = copy_data_N_2.copy() 
        copy_data_N_2_C.drop(copy_data_N_2_C.iloc[:, :min_index], inplace=True, axis=1)

        
    return First_n,copy_data_N_0_C,copy_data_N_1_C,copy_data_N_2_C, min_trialN


def snippet_fig2_3(df_allN,df_allN_smooth,neuron=0, class_labels = ['NaCl','CitricAcid','Quinine','Sucrose']):
    
    from Script.Script import get_spike_train_all,get_min_trial_numb
    import pandas as pd
    
    curr_n_ras = get_spike_train_all(df_allN, neuron)
    curr_n = get_spike_train_all(df_allN_smooth, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1

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

    if len(class_labels)==4:
        s_0 = pd.Series(unit_all[0])
        s_1 = pd.Series(unit_all[1])
        s_2 = pd.Series(unit_all[2])
        s_3 = pd.Series(unit_all[3])
        First_n = pd.Series(pd.concat((s_0,s_1,s_2,s_3),axis=0))
    
    min_index = df_allN_smooth.columns.get_loc('Trial') + 1
    copy_data = df_allN_smooth.copy()
    copy_data_N = copy_data[copy_data['Neuron']==neuron]

    if len(class_labels)==4:

        copy_data_N_0 = copy_data_N[copy_data_N['Taste']==0]
        copy_data_N_1 = copy_data_N[copy_data_N['Taste']==1]
        copy_data_N_2 = copy_data_N[copy_data_N['Taste']==2]
        copy_data_N_3 = copy_data_N[copy_data_N['Taste']==3]
        
        copy_data_N_0_C = copy_data_N_0.copy() 
        copy_data_N_0_C.drop(copy_data_N_0_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_1_C = copy_data_N_1.copy() 
        copy_data_N_1_C.drop(copy_data_N_1_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_2_C = copy_data_N_2.copy() 
        copy_data_N_2_C.drop(copy_data_N_2_C.iloc[:, :min_index], inplace=True, axis=1)
        
        copy_data_N_3_C = copy_data_N_3.copy() 
        copy_data_N_3_C.drop(copy_data_N_3_C.iloc[:, :min_index], inplace=True, axis=1)
        
        
    return First_n,copy_data_N_0_C,copy_data_N_1_C,copy_data_N_2_C,copy_data_N_3_C, min_trialN


def snippet_fig2_4(df_allN,df_allN_smooth,neuron=0):
    
    from Script.Script import get_spike_train_all,get_min_trial_numb
    import pandas as pd
    
    curr_n_ras = get_spike_train_all(df_allN, neuron)
    curr_n = get_spike_train_all(df_allN_smooth, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1

    class_labels = ['Cold','Hot']
    class_id     = [4,5]
    unit_all = []
    for tt in range(len(class_labels)): # change it to lenght of class_labels
        rast = curr_n_ras[curr_n_ras['Taste']== class_id[tt]].iloc[0:min_trialN,start_index:].reset_index(drop=True).T
        #rast.loc[rast.loc[0]]==1
        unit=[]
        for t in range(rast.shape[1]):
            temp = []
            for j in range(-2000, 2000, 1):
                if rast[t][j]>0:
                    temp.append(j)
            unit.append(temp)
        unit_all.append(unit)

    if len(class_labels)==2:
        s_0 = pd.Series(unit_all[0]) # Cold
        s_1 = pd.Series(unit_all[1]) # Hot
        First_n = pd.Series(pd.concat((s_0,s_1),axis=0))
    
    min_index = df_allN_smooth.columns.get_loc('Trial') + 1
    copy_data = df_allN_smooth.copy()
    copy_data_N = copy_data[copy_data['Neuron']==neuron]

    if len(class_labels)==2:

        copy_data_N_0 = copy_data_N[copy_data_N['Taste']==4]
        copy_data_N_1 = copy_data_N[copy_data_N['Taste']==5]
      
        
        copy_data_N_0_C = copy_data_N_0.copy() 
        copy_data_N_0_C.drop(copy_data_N_0_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_1_C = copy_data_N_1.copy() 
        copy_data_N_1_C.drop(copy_data_N_1_C.iloc[:, :min_index], inplace=True, axis=1)

      
    return First_n,copy_data_N_0_C,copy_data_N_1_C, min_trialN


def snippet_fig2_5(df_allN,df_allN_smooth,neuron=0):
    
    from Documents.GitHub.dataset_temp_gustatory.Script import get_spike_train_all,get_min_trial_numb
    import pandas as pd
    
    curr_n_ras = get_spike_train_all(df_allN, neuron)
    curr_n = get_spike_train_all(df_allN_smooth, neuron)
    min_trialN = get_min_trial_numb(curr_n)
    start_index = curr_n.columns.get_loc('Trial') + 1

    class_labels = ['Cold','Hot']
    class_id     = [2,3]
    unit_all = []
    for tt in range(len(class_labels)): # change it to lenght of class_labels
        rast = curr_n_ras[curr_n_ras['Taste']== class_id[tt]].iloc[0:min_trialN,start_index:].reset_index(drop=True).T
        #rast.loc[rast.loc[0]]==1
        unit=[]
        for t in range(rast.shape[1]):
            temp = []
            for j in range(-2000, 2000, 1):
                if rast[t][j]>0:
                    temp.append(j)
            unit.append(temp)
        unit_all.append(unit)

    if len(class_labels)==2:
        s_0 = pd.Series(unit_all[0]) # Cold
        s_1 = pd.Series(unit_all[1]) # Hot
        First_n = pd.Series(pd.concat((s_0,s_1),axis=0))
    
    min_index = df_allN_smooth.columns.get_loc('Trial') + 1
    copy_data = df_allN_smooth.copy()
    copy_data_N = copy_data[copy_data['Neuron']==neuron]

    if len(class_labels)==2:

        copy_data_N_0 = copy_data_N[copy_data_N['Taste']==2]
        copy_data_N_1 = copy_data_N[copy_data_N['Taste']==3]
      
        
        copy_data_N_0_C = copy_data_N_0.copy() 
        copy_data_N_0_C.drop(copy_data_N_0_C.iloc[:, :min_index], inplace=True, axis=1)

        copy_data_N_1_C = copy_data_N_1.copy() 
        copy_data_N_1_C.drop(copy_data_N_1_C.iloc[:, :min_index], inplace=True, axis=1)

      
    return First_n,copy_data_N_0_C,copy_data_N_1_C, min_trialN


def plot_r_and_psth_Lick(filePath,mouse,date,ras_start,ras_end,BinSize=0.075):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import os
    import pickle
    from tqdm import tqdm
    import time
    import seaborn as sns

    
    T = time.time()    
    font = {'family': 'arial',
            'color':  'black',
            'weight': 'normal',
            'size': 12,
            }
    start = 0
    end = 10000000
    fileName = str(filePath/mouse/date) + "//TsEvents/"

    dir_path = fileName
    # we read the lick time stamps
    file = 'TsEvents//Lick.csv'
    clus = pd.read_csv(filePath / mouse / date / file,header=None)
    clus = clus.to_numpy()
    clus = np.ravel(clus)

    # let's check how many events are present
    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith('TS.csv'):
            res.append(file)
    print(res)

    newpath =  str(str(filePath) + '\\' + mouse +  '\\' + date + '\\Licks_Analysis\\')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # save as a time vector for PSTHs plot as numpy vector
    t = np.arange(-ras_start, ras_end+BinSize, BinSize)
    Nam = 'Time_psth'
    np.save(newpath + Nam,t)  
        
    # generate a dictionary containing the timestamps of all the events as numpy arrays and save all the rasters as pickle and the PSTHs as numpy vectors
    fig = plt.figure()

    # from matplotlib import cm
    # n_colors = 10
    # colours = cm.rainbow(np.linspace(0, 1, n_colors))
    colours = sns.color_palette('rocket')
    BigRaster = pd.Series([],dtype='float64')

        
    BigColor = []
    ts_lick={}
    ts_lick['licks'] = clus
    ts_lick['Bin_psth'] = BinSize
    ts_lick['time_psth'] = t
    for ii in tqdm(range(len(res))):
        ts_lick[res[ii][:-4]] = np.ravel(pd.read_csv(dir_path + "/" + res[ii],header=None).to_numpy())
        ts_lick[res[ii][:-4] + "_raster"]={}
        for i in ts_lick[res[ii][:-4]]:
            temp = []
            for j in clus:
                if j > start:
                    if j < end:
                        if j > i-ras_start:
                            if j < i + ras_end:
                                temp.append(j-i)
            ts_lick[res[ii][:-4] + "_raster"][i] = temp       
        s = pd.Series(ts_lick[res[ii][:-4] + "_raster"])
        # start PSTHs
        template = np.zeros((len(s),len(t)-1),dtype = float)
        z = 0
        for iN in s.index:  # for all the trials
            v = np.array(s[iN])
            for tt in range(len(t)-1):
                TS = list(np.where(np.logical_and(v>=t[tt], v<=t[tt+1])))
                if TS[0].size==0:
                    template[z][tt] = 0
                else:
                    template[z][tt] = len(TS[0])/BinSize
            z=z+1
        ts_lick[res[ii][:-4] + "_psth"] = template
        #np.save(newpath + res[ii][:-4],template)      
        #s.to_pickle(newpath + res[ii][:-4]) # save the rasters 
        if ii == 0:
            BigRaster = s
            BigColor = [colours[ii+1]]*len(s)
        else:
            BigRaster = pd.concat([BigRaster, s],ignore_index=True)
            BigColor = BigColor + [colours[ii+1]]*len(s)

    plt.eventplot(BigRaster,color = BigColor)
    plt.ylabel('Trials',fontdict=font)
    plt.xlabel('Time (s)',fontdict=font)
    plt.xlim(-1*ras_start,ras_end)
    #figureid = str("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/Bouaichi_and_Vincis_jn_1B.pdf")
    #plt.savefig(figureid,transparent=True)
