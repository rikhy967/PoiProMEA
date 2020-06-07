#SORTING Ver2
import sys, importlib
from  McsPy.McsData import RawData
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import scipy
import pywt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D



def DetectSpike(segnale, soglia, fs, dead_time = 0.003):   
 
    """
    Detect spikes when the signal exceed in amplitude a certain threshold
    It works automatically with both positive and negativa thresholds
    
    PARAMETERS
    segnale = the signal in which it search for the peaks (it must be a numpy.array or a list or a tuple)
    soglia = the chosen thresold for the channel(it has to be int, float, double, numpy.float64)
    fs = sampling frequency (it must be an integer)
    dead_time [optional] = time in which the function doesn't search for a maximum after detecting one
    
    RETURN
    indici_spike[:k] = python list with length m, which contains all the samples of the signal when it cross the threshold
    
    """
    
    j = 0 
    numero_campioni = 0
    for j in segnale:
        numero_campioni += 1
    soglia = abs(soglia) #we consider both a positive and negative thresholds
    indici_spike = [None] * numero_campioni
    dead_campioni = int(dead_time*fs)
    i = 0
    k = 0
    while (i<numero_campioni):
        valore = abs(segnale[i])
        if (valore>soglia):
            indici_spike[k] = i
            k += 1
            i += dead_campioni
        else:
            i += 1
    return indici_spike[:k] 



def AlignSpike(segnale, indici, soglia, fs, research_time = 0.002):  
    """
    Align all the spikes previously detected with the function RilevaSpike
    
    PARAMETERS
    segnale = the signal to search the peaks in (it must be a numpy.array, or a list or a tuple)
    indici = the samples detected with the function RilevaSpike (they must be type integer)
    fs = sampling frequency (it must be type integer)
    research_timpe [optional] = time (in seconds) in which the function search for the relative maximum (default 0.002)
    
    RETURN
    indici_spike[:k] = a python list of length m, which contains all the samples of the spikes aligned to the minimum or maximum (if the signal excedees both, they are aligned to the minimum)
     
    """

    numero_campioni = len(segnale)
    research_campioni = int(research_time*fs)
    indici_allineati = [None] * numero_campioni
    soglia = abs(soglia)

    m=0
    for i in indici:
        k = 0
        picco_negativo = False  
        if (i + research_campioni) <= numero_campioni:
            while (k<research_campioni):   
                if segnale[i+k] < -soglia:
                    picco_negativo = True
                    indici_allineati[m] = i+k 
                    k+=1
                    break
                k+=1
            if picco_negativo == False:
                indici_allineati[m] = i 
                k=0 
                while (k<research_campioni):
                    if segnale[i+k] > segnale[indici_allineati[m]]:
                        indici_allineati[m] = i+k
                    k += 1     
            else:
                while (k<research_campioni):
                    if segnale[i+k] < segnale[indici_allineati[m]]:
                        indici_allineati[m] = i+k
                    k += 1
            m +=1    
        else:
            break
    return indici_allineati[:m]



def ExtractSpike(segnale, indici, fs, pre = 0.001, post = 0.002):
    """
    Extract the waveform of the spikes as an array
    
    PARAMETERS:
    segnale: the signal as an unidimensional numpy array 
    indice: the samples of the spikes, as a unidimensional numpy array
    pre: length of the cutoff in seconds before the spike
    post: length of the cutoff in seconds after the spike
    fs: sampling frequency
    
    RETURNS
    cutouts: bidimensional numpy array, with a spike in each row

    """

    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(indici)
    cutout = np.empty([lunghezza_indici, prima+dopo], np.int32)
 
    dim = segnale.shape[0]
    k=0
    for i in indici:
        #verifico che la finestra non esca dal segnale
        if (i-prima >= 0) and (i+dopo <= dim):
            cutout[k] = segnale[(int(i)-prima):(int(i)+dopo)]          
        k += 1
    return cutout


#____________________________________________________________________________________________PCA_________________________________________________


def EseguiPCA(dati, n=3, show=False):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    #Standardizing data
    standardizzati = StandardScaler().fit_transform(dati)
    print("\nSignal standardized\nMean: ", np.mean(standardizzati), "\nVariance: ", np.std(standardizzati)**2, "\n")

    #3D
    if n==3:
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(standardizzati)
        principal_DataFrame = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2','PC3'])

        if show == True:
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(principal_components[:,0], principal_components[:,1], principal_components[:,2], color='#000000', depthshade=True, lw=0)  
            plt.show()
        
    #2D    
    elif n==2:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(standardizzati)
        principal_DataFrame = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

        if show == True:
            fig = plt.figure(figsize = (15,15))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('Principal Component 1', fontsize = 30)
            ax.set_ylabel('Principal Component 2', fontsize = 30)

            x = principal_components[:,0]
            y = principal_components[:,1]
            ax.scatter(x, y, color ="#000000")
            ax.grid()
            plt.show()
    
    else:
        raise Exception("PCA funciton only work with 2 or 3 dimensions! n=2, n=3")
    
    return principal_DataFrame


#____________________________________________________________________________________________HIERARCHICAL_________________________________________________

def perform_pca_gerarchico(cutouts,spike_list,fs,n_comp=3, centroids=False):
    """
    
    1) Perform feature estraction using Principal Component Analysis (PCA) on cutouts (normalized).
    2) Perform Agglomerative clustering on the PCs
    
    Params:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster
     
    Return:
     - final_list: a list that contains an array of spikes for each cluster (neuron) 
    
    """
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    
    #Normalization
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    print('Total spikes: ', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    
    if len(spike_list) != transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]
    
    #List of the silhouette scores
    list_score = []   
    
    #Plot all graph, included the one with only one cluster (in this case without silhouette score)
    for n in range (1,6):
        model = AgglomerativeClustering(n_clusters=n, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
        cluster_labels = model.fit_predict(transformed)
        if(n!=1):
            silhouette_avg = silhouette_score(transformed, cluster_labels)
            print('\n______________________________________________________________________________________________________________')
            print("For", n,"cluster, the silhouette score is:", silhouette_avg)
            print('\n')
            list_score.append(silhouette_avg)

        #Plot PCA
        fig = plt.figure(figsize=(18,8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        color = []
        for i in cluster_labels:
            color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        if centroids == True:
            print(model.cluster_centers_)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],model.cluster_centers_[:,2],s=300,  c='black', depthshade=False)

        #Plot average waveforms
        ax = fig.add_subplot(1, 2, 2)
        for i in range(n):
            idx = cluster_labels == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cutouts[idx,:],axis = 0)
            std_wave = np.std(cutouts[idx,:],axis = 0)
            ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
            
        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        plt.show()  
         
    top_clusters = list_score.index(max(list_score))+2
    
    
    print("\n\n\033[1;31;47mBest cluster in the range 2 to 6: ",top_clusters,", with a silhouette score of: ",max(list_score), "\u001b[0m  \n\n")
    
    model = AgglomerativeClustering(n_clusters=top_clusters, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
    cluster_labels = model.fit_predict(transformed)
    print('Trans shape: ',transformed.shape)
    print('Spike list: ',len(spike_list))
    list_idx = list(np.unique(cluster_labels))
    
    final_list = []
    
    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]
    
    for i in list_idx:
        final_list.append(spike_list[cluster_labels==i])        
       
    return final_list

#____________________________________________________________________________________________GMIXTURE_________________________________________________

def perform_pca_gmixtures(cutouts, spike_list, fs, n_comp=3, centroids=False):
    """
    
    1) Perform feature estraction using Principal Component Analysis (PCA) on cutouts.
    2) Perform Gaussian Mixtures clustering on the PCs
    
    Params:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster
     
    Return:
     - final_list: a list that contains an array of spikes for each cluster (neuron) 
    
    """
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    
    #Normalizzo i dati
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    print('Total spikes', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    
    if len(spike_list) != transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]
    
    #List of the silhouette scores
    list_score = []   
    
    #Plot all graph, included the one with only one cluster (in this case without silhouette score)
    for n in range (1,6):
        model = GaussianMixture(n_components=n, max_iter=200, random_state=10, tol=0.0001)
        cluster_labels = model.fit_predict(transformed)
        if(n!=1):
            silhouette_avg = silhouette_score(transformed, cluster_labels)
            print('\n______________________________________________________________________________________________________________')
            print("For", n,"clusters, the silhouette score is:", silhouette_avg)
            print('\n')
            list_score.append(silhouette_avg)

        #Plot PCA
        fig = plt.figure(figsize=(18,8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        color = []
        for i in cluster_labels:
            color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        if centroids == True:
            print(model.cluster_centers_)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],model.cluster_centers_[:,2],s=300,  c='black', depthshade=False)

        #Plot average waveforms
        ax = fig.add_subplot(1, 2, 2)
        for i in range(n):
            idx = cluster_labels == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cutouts[idx,:],axis = 0)
            std_wave = np.std(cutouts[idx,:],axis = 0)
            ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
            
        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        plt.show()  
         
    top_clusters = list_score.index(max(list_score))+2
    
    print("\n\n\033[1;31;47mBest cluster in the range 2 to 6: ",top_clusters,", with a silhouette score of: ",max(list_score), "\u001b[0m  \n\n")
    
    model = GaussianMixture(n_components=top_clusters,max_iter=200,random_state=10, tol=0.0001)
    cluster_labels = model.fit_predict(transformed)
    print('Trans shape: ',transformed.shape)
    print('Spike list: ',len(spike_list))
    list_idx = list(np.unique(cluster_labels))
    
    final_list = []
    
    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]
    
    for i in list_idx:
        final_list.append(spike_list[cluster_labels==i])        
       
    return final_list


#____________________________________________________________________________________________KMEANS_________________________________________________

def perform_pca_kmeans(cutouts,spike_list,fs,n_comp, centroids=False):
    """
    
    1) Perform feature estraction using Principal Component Analysis (PCA) on cutouts.
    2) Perform Kmeans clustering on the PCs
    
    Params:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster
     
    Return:
     - final_list: a list that contains an array of spikes for each cluster (neuron) 
    
    """
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    
    #Normalizzo i dati
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    print('Total spikes', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
        
    if len(spike_list)!=transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]
    
    #Lista che contiene i silhouette score
    list_score = []  
    
        #Plotta tutti i grafici compreso quello con numero di cluster = 1 (in quel caso senza silhouette score)
    for n in range (1,6):
        model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=400, tol=0.00005, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')
        cluster_labels = model.fit_predict(transformed)
        if (n != 1):
            silhouette_avg = silhouette_score(transformed, cluster_labels)
            print('\n______________________________________________________________________________________________________________')
            print("For", n,"clusters, the silhouette score is:", silhouette_avg)
            print('\n')
            list_score.append(silhouette_avg)
        
        #Plot PCA
        fig = plt.figure(figsize=(18,8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        color = []
        for i in cluster_labels:
            color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        if centroids == True:
            print(model.cluster_centers_)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],model.cluster_centers_[:,2],s=300,  c='black', depthshade=False)

        #Plot average waveforms
        ax = fig.add_subplot(1, 2, 2)
        for i in range(n):
            idx = cluster_labels == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cutouts[idx,:],axis = 0)
            std_wave = np.std(cutouts[idx,:],axis = 0)
            ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
    
        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        plt.show()  
         
    top_clusters = list_score.index(max(list_score))+2
    
    print("\n\n\033[1;31;47mBest cluster in the range 2 to 6: ",top_clusters,", with a silhouette score of: ",max(list_score), "\u001b[0m  \n\n")
  
    model = KMeans(n_clusters=top_clusters, init='k-means++', n_init=10, max_iter=400, tol=0.00005, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
    cluster_labels = model.fit_predict(transformed)
    print('Trans shape: ',transformed.shape)
    print('Spike list: ',len(spike_list))
    list_idx = list(np.unique(cluster_labels))
    
    final_list = []
    
    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]
    
    for i in list_idx:
        final_list.append(spike_list[cluster_labels==i])

 

    return final_list

#____________________________________________________________________________________________DBSCAN_________________________________________________

def perform_pca_DBSCAN(cutouts, spike_list, fs, n_comp = 3, distanza = 1, punti_min = 15):
    """
    Params:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster
     
    Return:
     - final_list: a list that contains an array of spikes for each cluster (neuron) 
    
    """
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    import sklearn.preprocessing as ps

    scale = ps.StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    
    print('Total spikes', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    
    if len(spike_list)!=transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]
    
    list_score = []   # List that contains the results of the silhouette score for each cluster
    
    model = DBSCAN(eps=distanza, min_samples=punti_min, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=-1)
    cluster_labels = model.fit_predict(transformed)
    
    out=0
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    
    for i in range(cluster_labels.shape[0]):
        if (cluster_labels[i] == -1):
            out+=1
        elif (cluster_labels[i] == 0):
            a +=1
        elif (cluster_labels[i] == 1):
            b+=1        
        elif (cluster_labels[i] == 2):
            c+=1
        elif (cluster_labels[i] == 3):
            d+=1
        elif (cluster_labels[i] == 4):
            e+=1
        elif (cluster_labels[i] == 5):
            f+=1

    if(out!=0):
        print('\nSpike detected as noise', out)
    else:
        print('\nNo spike detected as noise')
        
    check=0
    
    color = []
    for i in cluster_labels:
        color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        if i==-1:
            color[i]='k'

    indici=cluster_labels
    coordinate=transformed  
    
    if(b!=0):
        for i in reversed(range(indici.shape[0])):
            if (indici[i] == -1):
                #print(coordinate[i])
                np.delete(coordinate, i)
                np.delete(indici, i)
            silhouette_avg = silhouette_score(coordinate, indici)
            print("\nNumber of clusters: ", len(set(cluster_labels))-1,"\nThe silhouette score is:", silhouette_avg)
            list_score.append(silhouette_avg)
            check=1
            break

    if check==0:
        print('\nOnly one cluster detected')
    if(a!=0):
        print('\nBlue spikes:', a)
    if(b!=0):
        print('\nOrange spikes:', b)
    if(c!=0):
        print('\nGreen spikes:', c)
    if(d!=0):
        print('\nRed spikes:', d)
    if(e!=0):
        print('\nPurple spikes:', e)
    if(f!=0):
        print('\nBrown spikes:', f)
    
    #Plot PCA 
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
    
    #Plot average waveform
    ax = fig.add_subplot(1, 2, 2)
    for i in range(len(set(cluster_labels))-1):
        idx = cluster_labels == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean_wave = np.mean(cutouts[idx,:],axis = 0)
        std_wave = np.std(cutouts[idx,:],axis = 0)
        ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
     
    plt.xlabel('Time [0.1ms]')
    plt.ylabel('Voltage [\u03BCV]')
    plt.show()    
        
    list_idx = list(np.unique(cluster_labels))
    final_list = []
    
    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]
    
    for i in list_idx:
        final_list.append(spike_list[cluster_labels==i])       
       
    return final_list

#____________________________________________________________________________________FILT BUTTERWORTH_________________________________

from scipy.signal import ellip, cheby1, bessel, butter, lfilter, filtfilt, iirfilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Implementation of Butterworth filtering.
    
    Params:
     - lowcut: low cutting frequency (Hz)
     - highcut: high cutting frequency (Hz)
     - fs: sampling frequency (Hz)
     - order: order of the filter
    
    Return:
     - b,a = coefficients of the filter
    
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Perfom the filtering of the data using a zero-phase Butterworth filter.
    
    Params:
     - data: The signal as a 1-dimensional numpy array
     - lowcut: low cutting frequency (Hz)
     - highcut: high cutting frequency (Hz)
     - fs: sampling frequency (Hz)
     - order: order of the filter
     
    Return:
     - y = signal filtered
    
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


