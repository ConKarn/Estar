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

#####  Otsuthresh // Cut


def Otsuthresh(Map,Obs,NanMap=None,output="thresh", savepath=""):
    Crbin = Map.copy()
    validpoints=(Crbin[Obs>=1]).flatten()
    validpoints=validpoints[~np.isnan(validpoints)]
    threshotsu = threshold_otsu(validpoints)
    Crbin[Crbin<threshotsu]=0
    Crbin[Crbin>=threshotsu]=1
    x,y = np.where(Obs>=1)
    plt.figure(figsize=(10,10))
    colors=['grey','lightgreen']
    cmap = ListedColormap(colors)
    if NanMap is not None:
        Crbin=Crbin.astype("float64")
        Crbin[NanMap]=None
    plt.imshow(Crbin,cmap=cmap)
    cbar=plt.colorbar(shrink=0.7,ticks=np.arange(len(colors)))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Likely absence', 'Likely presence'])  # Définir les étiquettes des ticks
    plt.scatter(y,x,c="red",s=0.1,label="Observation points")
    plt.legend()
    
    if savepath != "":
        print("Saving plot...")
        plt.savefig(savepath,dpi=300)
        #plt.show(block=False)
    ##plt.show(block=False)
    if output=="thresh":
        return threshotsu
    else:
        return Crbin
    
    
def Cut(Map,Obs,thresh=0.5,NanMap=None,output="thresh", savepath=""):
    Crbin = Map.copy()
    Crbin[Crbin<thresh]=0
    Crbin[Crbin>=thresh]=1
    x,y = np.where(Obs>=1)
    plt.figure(figsize=(10,10))
    colors=['grey','lightgreen']
    cmap = ListedColormap(colors)
    if NanMap is not None:
        Crbin=Crbin.astype("float64")
        Crbin[NanMap]=None
    plt.imshow(Crbin,cmap=cmap)
    cbar=plt.colorbar(shrink=0.7,ticks=np.arange(len(colors)))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Likely absence', 'Likely presence'])  # Définir les étiquettes des ticks
    plt.scatter(y,x,c="red",s=0.1,label="Observation points")
    plt.legend()
    
    if savepath != "":
        print("Saving plot...")
        plt.savefig(savepath,dpi=300)
        #plt.show(block=False)
    ##plt.show(block=False)
    if output=="thresh":
        return thresh
    else:
        return Crbin
    


def Find_focal_area(Obs,NanMap,divgrid=5,plotgrid=False):
    
    count=0
    if type(divgrid)==int:
        Lw=[]
        Lextent=[]

        data=Obs>=1

        # Diviser le tableau en 25 morceaux et trouver le rectangle avec la somme de valeurs la plus grande
        max_sum = float('-inf')
        max_sum_rect = None

        # Diviser les deux axes en 5 parties égales
        subshape = (data.shape[0] // divgrid, data.shape[1] // divgrid)
        xobs,yobs=np.where(Obs>=1)
        N=len(xobs)
        
        if plotgrid==True:
            plt.figure(figsize=(10,10))
            plt.imshow(1-NanMap,cmap=ListedColormap(["white","grey"]))
            plt.scatter(yobs,xobs,s=0.1,c="lightgreen")

        for i in range(divgrid):
            for j in range(divgrid):
                # Calculer les indices de début et de fin pour chaque sous-ensemble
                start_row, start_col = i * subshape[0], j * subshape[1]
                end_row, end_col = start_row + subshape[0], start_col + subshape[1]

                # Extraire le sous-ensemble de données
                sub_data = data[start_row:end_row, start_col:end_col]

                # Calculer la somme des valeurs dans le sous-ensemble
                sub_sum = np.nansum(sub_data)
                weight=np.log(1+sub_sum)
                Lw.append(weight)
                Lextent.append((start_row,end_row,start_col,end_col))

                width=end_col-start_col
                height=end_row-start_row
                (y0,y1,x0,x1)=(start_row,end_row,start_col,end_col)
                if plotgrid==True:
                    plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],c="tomato")
                    #plt.fill([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], color='red')
                    plt.text(1/2*(x0+x1),1/2*(y0+y1),str(int(sub_sum))+" obs",ha="center",va="center")
                if int(sub_sum)>30:
                    count+=1
        #if plotgrid==True:
            ##plt.show(block=False)
        Lw=np.array(Lw)
        Lw=Lw/Lw.sum()
    else:
        if plotgrid==True:
            plt.figure(figsize=(10,10))
            plt.imshow(1-NanMap,cmap=ListedColormap(["white","grey"]))
        xobs,yobs=np.where(Obs>=1)
        N=len(xobs)
        if plotgrid==True:
            plt.scatter(yobs,xobs,s=0.1,c="lightgreen")
        data=Obs>=1
        Lextent=[]
        start_row,end_row,start_col,end_col=divgrid
        sub_data = data[start_row:end_row, start_col:end_col]
        (y0,y1,x0,x1)=(start_row,end_row,start_col,end_col)
        if plotgrid==True:
            plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],c="tomato")
        #plt.fill([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], color='red')
        sub_sum=np.nansum(sub_data)
        if plotgrid==True:
            plt.text(1/2*(x0+x1),1/2*(y0+y1),str(int(sub_sum))+" obs",ha="center",va="center")
        if int(sub_sum)>30:
            count+=1
        Xtend=(start_row,end_row,start_col,end_col)
        Lextent.append(Xtend)
        Lw=[1]
    print("divgrid=",divgrid,"count=",count)
    return Lw,Lextent,count


# on peut calculer Boyce sur tous les divisions de grilles avec des observations, puis pondérer les résultats par le nombre
# d'observations dans ces grilles....

#pour chaque divisiion de grille calculer Boyce Puis faire la moyenne pondérée de toutes les courbes....

def BoyceIndexTresh(CR,Obs,NanMap,plot=False,plotFourier=False,HSxIUCN=None,save="",path=None,thresh_default=0.5):
    # Maintenant que les groupes sont créés on va compter le nombre d'observations tombant dans la range estimée en fonction 
    # de l'habitat suitability (HS) prédite, et cela en fonction de l'aire sur la carte concernant cet intervalle d'HS
    
    Obs_c = Obs.copy()
    Obs_c[Obs_c>1]=1
    NObs = np.nansum(Obs_c)
    print("spname =",save,"N Obs = ",NObs)
    if NObs<30:
        print("/!\ Not enough observations for a proper trinarisation using BoyceIndex /!")
        print("Binarisation using Otsu thresholding instead .... ")
        points = np.where(Obs_c>=1)
        CR_values_at_points = CR[points]
        thresh_CR = threshold_otsu(CR_values_at_points)
        print("Successful binarisation, chosen threshold= ",thresh_CR)
        CRbin=CR.copy()
        CRbin[np.isnan(CR)==False]=0
        CRbin[CR==thresh_CR]=0.5
        CRbin[CR>thresh_CR]=1
        if plot==True:
            plt.figure()
            plt.xlabel("X (km)")
            plt.ylabel("Y (km)")
            plt.title("Estimated Current Range binary "+save)
            colors=['grey','darkgrey','lightgreen']
            cmap = ListedColormap(colors)
            plt.imshow(CRbin,cmap=cmap)
            cbar=plt.colorbar(shrink=0.7,ticks=np.arange(len(colors)))
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['Likely absence', 'Unsure', 'Likely presence'])  # Définir les étiquettes des ticks
            y,x = np.where(Obs==1)
            plt.scatter(x,y,c="red",s=0.1,label="Observation points")
            plt.legend()
            ##plt.show(block=False)

        if save!="":
            plt.savefig(path+"/plots/"+save+'.png', dpi=200)
            plt.close()
        
        return CRbin,thresh_CR
            
    
    def boyce(model,plotfis=False):
        ndivopt=0
        loop=True
        while loop==True:
            ndivopt+=1
            if ndivopt==11:
                loop=False
                ndivopt=1
            Lw,Lextent,ndivi=Find_focal_area(Obs,NanMap,divgrid=int(ndivopt),plotgrid=plotfis)
            lFi=[]
            lFw=[]
            for rectangle,w in zip(Lextent,Lw):
                (r1,r2,c1,c2)=rectangle
                CRi=model[r1:r2,c1:c2] # focal submap
                Obsi=Obs[r1:r2,c1:c2]
                Obsi[Obsi>1]=1
                N=np.nansum(Obsi)
                #print("N=",N)
                if N>=30: # if there is enough observations to deduce a correct BoyceIndex curve
                    AtLeastOneSquare=True
                    plt.figure()
                    plt.close()
                    # for each model used to plot (default = P(Ps|E*s) vs HS )
                    print("Computing curves")
                    t1=Time.time()
                    Fis=[]
                    Extentmax=np.nansum(np.isnan(CRi)==False)
                    for a in np.arange(0.05,1.05,0.1):
                        t2=Time.time()
                        if t2-t1>5:
                            print(a*100,"%")
                            t1=Time.time()
                        Extent_i=(CRi >= a-0.05)*(CRi<= a+0.05)
                        P=np.nansum(Obsi[Extent_i])
                        E=(Extent_i.sum()/Extentmax)*N
                        if P==0:
                            Fis.append(0.0)
                        else:
                            Fis.append(P/E)

                    corrspearman=spearmanr(np.arange(0.05,1.05,0.1),Fis)[0]
                    print("correlation de spearman = ",corrspearman)
                    maxFi=max(Fis)
                    print("max Fi = ",maxFi)
                    if (corrspearman>0.8 and maxFi>2) or (loop==False and len(lFi)==0):
                        loop=False
                        lFw.append(w)
                        lFi.append(Fis)
                        #if plotfis==True:
                        #    plt.plot(np.arange(0.05,1.05,0.1),Fis,label="Calibration curve for the specified rectangle")
                        #    plt.title("F score")
                        #    plt.xlabel("HS")
                        #    plt.ylabel("Fi")
                        #    plt.plot([0,1],[1,1],linestyle="--",color="red")
                        #    plt.legend()
                        #    ##plt.show(block=False)
        return [lFi,lFw,ndivopt]
    
    lFi,lFw,ndivopt = boyce(CR,plotfis=plot)
    
    if HSxIUCN is not None: # if we compare it to HSxIUCN
        lFi_hsxiucn , lFw_hsxiucn, ndivopt_hsxiucn = boyce(HSxIUCN,plotfis=False)

    plt.figure(figsize=(7,7))
    plt.xlabel("P(Ps=ps) from model")
    plt.ylabel("Observed points / Expected points")
    
    def plotlFilFw(lFi,lFw,ndivopt,name,color,plotbar=True):

        WeightedMean=np.array([0.0]*len(lFi[0]))
        for fis,w in zip(lFi,lFw):
            if plotbar==True:
                plt.plot(np.arange(0.05,1.05,0.1),fis,color=color,alpha=0.08,linewidth=w/(1/(ndivopt**2)))
            WeightedMean+=np.array(fis)*w
        WeightedMean=WeightedMean/sum(lFw)

        # for each column of var compute wilcoxon test compared to 1
        frstidxequals0=0
        lastidxequals0=0
        frst=True
        for idx in range(len(lFi[0])):
            wm=WeightedMean[idx]
            if frst==True and wm>=0.5:
                frst=False
                frstidxequals0=idx
            if wm<=1.5:
                lastidxequals0=idx
        if plotbar==True:
            plt.plot(np.arange(0.05,1.05,0.1),WeightedMean,color=color,alpha=1,label="Calibration curve: model="+name)
        X1=np.arange(0.05,1.05,0.1)[frstidxequals0]
        X2=np.arange(0.05,1.05,0.1)[lastidxequals0]
        if plotbar==True:
            plt.fill_between(x=np.linspace(0,X1,50),y1=[0]*50,y2=[1]*50,color="grey")
            plt.fill_between(x=np.linspace(X1,X2,50),y1=[0]*50,y2=[1]*50,color="darkgrey")
            plt.fill_between(x=np.linspace(X2,1,50),y1=[0]*50,y2=[1]*50,color="lightgreen")
            plt.plot([0,1],[1,1],linestyle='--',color="black")
        return [X1,X2]
    
    X1,X2=plotlFilFw(lFi,lFw,ndivopt,"CR","blue",plotbar=plot)
    if HSxIUCN is not None:
        if plot==True:
            plotlFilFw(lFi_hsxiucn , lFw_hsxiucn,ndivopt_hsxiucn,"HSxIUCN","orange",plotbar=False)

    CRbin=CR.copy()
    CRbin[np.isnan(CR)==False]=0
    CRbin[CR>X1]=0.5
    CRbin[CR>X2]=1

    
    plt.grid()
    plt.legend()
    ##plt.show(block=False)
    plt.figure()
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.title("Estimated Current Range binary "+save)
    colors=['grey','darkgrey','lightgreen']
    cmap = ListedColormap(colors)
    plt.imshow(CRbin,cmap=cmap)
    cbar=plt.colorbar(shrink=0.7,ticks=np.arange(len(colors)))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Likely absence', 'Unsure', 'Likely presence'])  # Définir les étiquettes des ticks
    y,x = np.where(Obs==1)
    plt.scatter(x,y,c="red",s=0.1,label="Observation points")
    plt.legend()
    if save!="":
        plt.savefig(path+"/plots/"+save+'.png', dpi=200)
    plt.close()
    CR_thresh = (1/2) * (X1 + X2) # cut the unsure region in half
    return CRbin, CR_thresh


print("Binarisation methods imported")