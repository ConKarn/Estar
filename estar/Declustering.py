
from tifffile import imsave
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time as Time
import scipy.stats
from scipy import stats
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from scipy import stats
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
import random as rd 
from scipy.signal import fftconvolve
from scipy import ndimage
from rasterio.transform import from_origin
from pyproj import Proj, transform
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import label, distance_transform_edt, sobel
from scipy.special import expit  # Pour la fonction logistique
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import matplotlib.pyplot as plt
from collections import Counter
import random
from .utils import* 
from .validation_tools import*



def generate_full_grid(large_xsize,large_ysize,cellsize):
    Xs=large_xsize
    Ys=large_ysize
    nbcelly = Ys//cellsize
    rest = Ys-cellsize*nbcelly
    Ygrid = np.repeat(np.arange(nbcelly),cellsize)
    Ygrid= np.concatenate((Ygrid, np.repeat(nbcelly+1,rest)))
    Ygrid = np.tile(Ygrid, Xs)
    Ygrid=Ygrid.reshape(Xs,Ys)
    # for X now
    nbcellx = Xs // cellsize
    restx = Xs - nbcellx*cellsize
    Xgrid = np.repeat(np.arange(nbcellx) * nbcelly, Ys * cellsize)   
    Xgrid = np.concatenate((Xgrid,np.repeat(nbcellx*nbcelly + nbcelly ,Ys*restx))) 
    Xgrid=Xgrid.reshape(Xs,Ys)
    large_grid = Xgrid+Ygrid
    return large_grid

def extract_sub_grid(large_grid, xsize, ysize, cellsize):

    # Choisir une position aléatoire pour extraire le sous-tableau
    start_x = rd.randint(0, cellsize)
    start_y = rd.randint(0, cellsize)

    # Extraire un sous-tableau de la taille demandée
    sub_grid = large_grid[start_x:start_x + xsize, start_y:start_y + ysize]
    
    return sub_grid

def generate_grid(xsize, ysize, cellsize):
    # Créer un tableau plus grand que nécessaire pour éviter les chevauchements
    # Ajouter cellsize pour avoir de la marge pour l'extraction
    large_xsize = xsize + cellsize
    large_ysize = ysize + cellsize

    # Générer la grille complète
    large_grid = generate_full_grid(large_xsize, large_ysize, cellsize)
    
    # Extraire le sous-tableau avec une position aléatoire
    grid = extract_sub_grid(large_grid, xsize, ysize, cellsize)
    return grid

# this function creates a debiased map by downweighting observations points in high density clusters to avoid 
# over sampling discrepancies in the distribution model.

def celld(Obs,NanMap,sig,gridcellsize=10,plotgrid=True,mode="diagnose",listdivgrids=None,background=None,by_declust=1,returncellsize=False,plot=False,Ksig=None):
    if Ksig is not None:
        print("specific kernel provided...")
        K=Ksig
    else:
        K=gkern(sig=sig,hole=True)
    RefZ = applykernel_pr(Obs,K) # raw density before declusterisation
    MeanZ = np.sum(RefZ[Obs==1]) / np.sum(Obs==1)
    listsd4plot=[]
    print("Non declustered mean =", MeanZ)
    
    for k in range(2): # first to find the optimal cell size, and then to compute the declustered map
        if k==0:
            mode="diagnose"
            print("diagnose")
        if k==1:
            mode="active"
            print("active")
        
        if mode=="diagnose":
            listmeans=[] # list of different means for the different cells
        weighted_obs_map = Obs*0.

        if NanMap is None:
            NanMap= np.zeros_like(Obs)
        count=0
        if mode=="active" and type(gridcellsize)==int:
            listdivgrids=[gridcellsize]
        if mode=="diagnose":
            if listdivgrids is None:
                listdivgrids=np.arange(1,30,1) # default 
                # on part de 1 jusqu'à obtenir un max, puis continuer jusqu'à 10 fois sans nouveau max 
        
        minMean = float('+inf')
        nb_step_with_same_min=0
        divgrid=1
        while nb_step_with_same_min <= 10 and divgrid < min(Obs.shape[0],Obs.shape[1])//2:
                    
            if mode=="diagnose":
                counts=[]
                divgrid+=by_declust
                
            if mode=="active":
                divgrid = gridcellsize

            data=Obs>=1
            t0=Time.time()
            
            mode_declust = "FastLarge"
            
            if mode_declust == "FastLarge":
                # Diviser le tableau en 25 morceaux et trouver le rectangle avec la somme de valeurs la plus grande
                max_sum = float('-inf')
                max_sum_rect = None

                # Diviser les deux axes en 5 parties égales
                xshape , yshape = data.shape
                nbdivx = xshape // divgrid
                nbdivy = yshape // divgrid
                subshape = ( divgrid,divgrid)
                xrange4shift = xshape - divgrid*nbdivx
                yrange4shift = yshape - divgrid*nbdivy
                #print("rangeshift=",xrange4shift)
                xobs,yobs=np.where(Obs>=1)
                N=len(xobs)
                nmax=5
                print("cell size = "+ str(divgrid)+"// ",end=" ")
                
                recup_diff_means=[]
                
                for n in range(nmax):
                    
                    partial_counts=[]
                    
                    print(n/nmax*100,"%",end=" ")
                    shiftx = int(rd.random()*divgrid)
                    shifty = int(rd.random()*divgrid)
                    #print(shiftx,shifty)

                    #if plotgrid==True and mode=="active":
            #             plt.figure(figsize=(10,10))
            #             plt.imshow(1-NanMap,cmap=ListedColormap(["white","grey"]))
            #             plt.scatter(yobs,xobs,s=0.1,c="green")
                    nbocc=0

                    for i in range(nbdivx):
                        for j in range(nbdivy):
                            # Calculer les indices de début et de fin pour chaque sous-ensemble
                            start_row, start_col = i * subshape[0] + shiftx, j * subshape[1] + shifty
                            end_row, end_col = start_row + subshape[0] + shiftx , start_col + subshape[1] + shifty

                            # Extraire le sous-ensemble de données
                            sub_data = data[start_row:end_row, start_col:end_col]
                            sub_Z = RefZ[start_row:end_row, start_col:end_col]
                            
                            xpoints,ypoints = np.where(sub_data==1) 
                            # Calculer la somme des valeurs dans le sous-ensemble
                            sub_sum = np.nansum(sub_data)
                            if mode=="diagnose":
                                if sub_sum !=0:
                                    mean_cell = 1/sub_sum * np.sum(sub_data*sub_Z) # moyenne de la cellule
                                    #counts . append(mean_cell)
                                    partial_counts.append(mean_cell) 
                                    nbocc+=1
                            if mode == "active":
                                if sub_sum >1 : # we discard observations alone in their cell (too much isolated)
                                    # all points in that region get a weight inversed corresponding to the number of points
                                    weighted_subdata = 1/sub_sum * sub_data * 1/nmax * (divgrid)**2 # approximately the partial sum observed the computation of the mean area associated per point
                                    weighted_obs_map[start_row:end_row,start_col:end_col] += weighted_subdata
                                elif sub_sum ==0:
                                    weighted_subdata = sub_data*0.
                                    weighted_obs_map[start_row:end_row,start_col:end_col] += weighted_subdata
                    recup_diff_means.append(np.mean(partial_counts))  # moyenne sur la carte pour cette grille
                #counts = np.array(counts)
                recup_diff_means = np.array(recup_diff_means)
            
            if mode=="active":
                break
            if mode=="diagnose":
                mean= np.mean(recup_diff_means) # moyenne des cartes pour les différentes grilles
                listmeans.append(mean)
                sd=np.var(recup_diff_means)**(1/2) # ecart type sur les moyennes calculées selon la grille
                listsd4plot.append(sd)
            
            current_min = np.min(listmeans)
            if current_min < minMean:
                minMean = current_min
                nb_step_with_same_min = 0
            else:
                nb_step_with_same_min += 1
            if Time.time() - t0 > 5:
                print("#cur_mean="+str(round(current_min,3))+"#min_mean="+str(round(minMean,3))+"#nb_step_with_same_mean="+str(nb_step_with_same_min),end=" ")
                t0 = Time.time()

        if mode=="diagnose":
            print("diagnosis >")
            mini = np.arange(1,divgrid+by_declust,by_declust)[listmeans.index(min(listmeans))+1]
            print("mini ok")
            if plot==True:
                plt.figure(figsize=(10,4))
                plt.grid()
                plt.title("Declustered Mean ~ Size of grid cells")
                plt.xlabel("Width of cells used for the grid (pixels)")
                plt.ylabel("Declustered Mean")
                ycurve = np.array([MeanZ]+listmeans)
                sigcurve=np.array( [0] + listsd4plot)/2
                plt.scatter(np.arange(1,divgrid+by_declust,by_declust),ycurve)
                plt.plot(np.arange(1,divgrid+by_declust,by_declust),ycurve,linewidth=2,label="Declustered Mean density")
                plt.plot([mini,mini],[0,min(listmeans)],linestyle='--',color="red",linewidth=2)
                plt.plot([1,divgrid],[MeanZ,MeanZ],linestyle="--",color="darkblue",linewidth=2)
                plt.text(divgrid//2,(1+ 1/20)*MeanZ, 'Non Declustered Mean density', fontsize=12, color='darkblue',horizontalalignment="center")
                plt.fill_between(np.arange(1,divgrid+by_declust,by_declust),ycurve-sigcurve, ycurve + sigcurve, color='lightblue', alpha=0.5)
                plt.text(mini + divgrid/50,min(listmeans)/2,"Optimal cell size",fontsize=12,color="red")
                plt.ylim(0,(1+ 1/5)*MeanZ)
                plt.legend()
                plt.show()


        if mode=="active":           
            ys,xs=np.where(weighted_obs_map!=0)
            ws=weighted_obs_map[weighted_obs_map !=0]
#             ws = ws/np.max(ws)
#             ws=ws*100
            if plot==True:
                plt.figure(figsize=(10,10))
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("Declustered Map")
                ws4plot = ws.copy()
                ws4plot /= np.max(ws4plot)
                ws4plot*=100
                if background is not None:
                    plt.imshow(background)
                    plt.scatter(xs,ys,s=ws4plot,c=ws4plot,cmap="magma",label="Observation points")
                    cbar = plt.colorbar(shrink=0.6)
                    cbar.set_label('Assigned weights')
                if background is None:
                    plt.scatter(ys,xs,s=ws4plot,c=ws4plot,cmap="magma",label="Observation points")
                plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
                plt.legend()
                plt.show()
    
        gridcellsize = int(mini+1)
    if returncellsize==True:
        return gridcellsize
    return (weighted_obs_map/(2*np.pi))**(1/2)


def Density_declustered (declustered_weights):
    declust = declustered_weights
    dun= np.unique(declust)
    Nb_group_ksize = 10
    dun = plt.hist(dun,bins=10)
    boundaries_ksize = list(dun[1])
    imax,jmax = declust.shape
    finalMap = np.zeros((imax,jmax))
    for k in range(len(boundaries_ksize)-1):
        b1 = boundaries_ksize[k] ; b2 = boundaries_ksize[k+1]
        declust_filtered = declust*(declust<=b2)*(declust>b1)
        partialMap = declust_filtered!=0
        sigma = int(1/2*(b1+b2)) # mean between boundaries
        if sigma ==0:
            sigma+=1
        print("sigma=",int(sigma))
        kernel = gkern(sig=sigma)
        kernel /= np.max(kernel)
        partialMap = applykernel_pr(partialMap,kernel) 
        finalMap+= partialMap
        
    return finalMap

# this function aims to deduce a cutoff threshold for currrange output based on calibration curve
print("Declustering methods imported")