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
from .Declustering import*
from .validation_tools import*
from .binarisation_methods import*

t0=Time.time()
S=0

def minimum_convex_polygon(array_2d):

    '''Calculate the minimum convex polygon that encloses all points with value 1 in a 2D array.'''

    # Get the coordinates of points with value 1
    points = np.column_stack(np.where(array_2d == 1))
    #print(points)
    # If there are fewer than 3 points, a convex hull can't be formed
    if points.shape[0] < 3:
        return array_2d
    
    # Calculate the convex hull of the points
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Create a mask array with the same shape as input
    mask = np.zeros_like(array_2d)
    
    # Use skimage's polygon function to fill the convex hull area
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], mask.shape)
    mask[rr, cc] = 1  # Set the region within the polygon to 1
    
    return mask

def MCP(Obs_raw,HS):

    '''Compute the minimum convex polygons for the given observations.'''

    Obs=Obs_raw.copy()
    Obs[Obs>=1]=1
    print(np.unique(Obs))
    finalMCP=np.zeros_like(Obs)
    x,y=np.where(Obs==1)
    values=HS[Obs==1]
    T=threshold_otsu(values)
    print("threshold_otsu=",T)
    highHS = HS>T
    patches = label(highHS, structure=np.ones((3, 3)))[0]
    plt.figure()
    plt.title("HS decomposition")
    plt.scatter(y,x,color="red",s=10,alpha=0.5)
    plt.imshow(patches)
    #plt.show(block=False)
    print("nb patches = ",len(np.unique(patches)))
    # filter all patches without obs
    labels = patches[Obs==1]
    print(labels)
    counts = Counter(labels)
    filtered_list = [item for item in labels if counts[item] >= 3]
    labels=np.unique(filtered_list)    
    print("iD with obs= ",labels)

    print("nb patches with obs =", len(labels))
    t0 = Time.time()
    n=0
    for iD in labels:
        n+=1
        if Time.time()-t0 >5:
            print(n/len(labels)*100,"% done")
            t0=Time.time()
        if iD != 0:
            patch=patches==iD
            obs_patch=Obs*patch
            nbpoints=np.nansum(obs_patch)
            if nbpoints>=3:
                try:
                    mcp = minimum_convex_polygon(obs_patch)
                    finalMCP+= mcp
                except Exception as e:
                    print("ConvexHull failed: not enough points, and too close, abort this region containing ",nbpoints," points")
    finalMCP[finalMCP>1]=1
    plt.figure()
    plt.title("Minimum convex polygons")
    plt.imshow(finalMCP)
    #plt.show(block=False)
    return finalMCP
    

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.filters import threshold_otsu
from scipy.spatial.distance import cdist


from scipy.spatial import KDTree

def compute_first_quartile_distance(S0, S1):

    '''Calculate the first quartile of the minimum distances from points in S0 to points in S1.'''

    # Get coordinates of points in S0 and S1
    S0_points = np.column_stack(np.where(S0 == 1))
    S1_points = np.column_stack(np.where(S1 == 1))
    
    # Build KDTree for S1 points
    tree = KDTree(S1_points)
    
    # Query the nearest S1 point for each S0 point
    min_distances, _ = tree.query(S0_points, k=1)
    
    # Calculate the first quartile (25th percentile) of the minimum distances
    first_quartile = np.percentile(min_distances, 25)
    
    return first_quartile


def OBRmap(Obs, HS):

    '''Compute the OBR map based on observations and habitat suitability.'''

    # Étape 1 : Calcul de la distance maximale minimale entre les points d'observations
    print("Calcul nb Obs...")
    obs_points = np.argwhere(Obs == 1)
    if len(obs_points) < 2:
        raise ValueError("Il doit y avoir au moins deux points d'observation pour calculer T.")
    print("Computing pairwise distances")
    # Calcul des distances entre tous les points d'observation
    distances = cdist(obs_points, obs_points)
    np.fill_diagonal(distances, np.inf)  # Ignorer la distance de chaque point avec lui-même
    print("finding maximal minimal distance...")
    T = np.min(distances, axis=1).max()  # Plus grande distance minimale
    print("T=",T)
    # Étape 2 : Seuil d'Otsu pour binariser HS en fonction des valeurs aux points d'observations
    hs_values_at_obs = HS[Obs == 1]
    threshold = threshold_otsu(hs_values_at_obs)
    print("Otsu threshold=",threshold)
    HS_binary = (HS > threshold).astype(int)
    plt.figure()
    plt.imshow(HS_binary,cmap="binary")
    #plt.show(block=False)
    # Étape 3 : Identification des régions continues de "1" (S1 et S0)
    labeled_HS, num_features = label(HS_binary)
    
    # Déterminer les régions contenant des observations (S1)
    S1_regions = set(labeled_HS[Obs == 1]) - {0}  # 0 représente l'arrière-plan
    
    # Générer la carte binaire pour les régions contenant des observations (S1)
    S1_mask = np.isin(labeled_HS, list(S1_regions)).astype(int)
    
    plt.figure()
    plt.title("S1 Regions")
    plt.imshow(S1_mask,cmap="binary")
    #plt.show(block=False)
    
    # Étape 4 : Identification des régions S0 proches des régions S1
    print("Compute S0 regions...")
    distance_to_S1 = distance_transform_edt(1 - S1_mask)
    S0_mask = ((distance_to_S1 > 0) & (distance_to_S1 <= T) & (HS_binary == 1)).astype(int)
    
    plt.figure()
    plt.title("All S0 Regions")
    plt.imshow(S0_mask,cmap="binary")
    #plt.show(block=False)
    
    result = (S1_mask | S0_mask).astype(int)
    
    plt.figure()
    plt.title("OBR selection")
    plt.imshow(result,cmap="binary")
    #plt.show(block=False)

    return result

def LRmap(Obs, HS):

    '''Compute the LR map based on observations and habitat suitability.'''

    # Étape 1 : Calcul de la distance maximale minimale entre les points d'observations
    obs_points = np.argwhere(Obs == 1)
    if len(obs_points) < 2:
        raise ValueError("Il doit y avoir au moins deux points d'observation pour calculer T.")
    
    # Calcul des distances entre tous les points d'observation
    distances = cdist(obs_points, obs_points)
    np.fill_diagonal(distances, np.inf)  # Ignorer la distance de chaque point avec lui-même
    T = np.min(distances, axis=1).max()  # Plus grande distance minimale
    print("T=",T)
    # Étape 2 : Seuil d'Otsu pour binariser HS en fonction des valeurs aux points d'observations
    hs_values_at_obs = HS[Obs == 1]
    threshold = threshold_otsu(hs_values_at_obs)
    print("Otsu threshold=",threshold)
    HS_binary = (HS > threshold).astype(int)
    plt.figure()
    plt.imshow(HS_binary,cmap="binary")
    #plt.show(block=False)
    # Étape 3 : Identification des régions continues de "1" (S1 et S0)
    labeled_HS, num_features = label(HS_binary)
    
    # Déterminer les régions contenant des observations (S1)
    S1_regions = set(labeled_HS[Obs == 1]) - {0}  # 0 représente l'arrière-plan
    
    # Générer la carte binaire pour les régions contenant des observations (S1)
    S1_mask = np.isin(labeled_HS, list(S1_regions)).astype(int)
    
    plt.figure()
    plt.title("S1 Regions")
    plt.imshow(S1_mask,cmap="binary")
    #plt.show(block=False)
    
    # Étape 4 : Identification des régions S0 proches des régions S1
    print("Compute S0 regions...")
    distance_to_S1 = distance_transform_edt(1 - S1_mask)
    print("Compute Lower Quantile distance")
    
    allRegions = HS_binary
    S0_mask_raw = allRegions - S1_mask
    first_quartile_distance = compute_first_quartile_distance(S0_mask_raw, S1_mask)
    print("First quartile of minimum distances:", first_quartile_distance)
    
    print("Select S0 regions...")
    S0_mask = ((distance_to_S1 > 0) & (distance_to_S1 <= first_quartile_distance) & (HS_binary == 1)).astype(int)
    
    plt.figure()
    plt.title("All selected S0 Regions")
    plt.imshow((S0_mask_raw + S0_mask),cmap="viridis")
    #plt.show(block=False)
    
    result = (S1_mask | S0_mask).astype(int)
    
    plt.figure()
    plt.title("LR selection")
    plt.imshow(result,cmap="binary")
    #plt.show(block=False)

    return result


def PseudoRange(SubRegions,Obs,Obstaxa=None,plot=False,NanMap=None,weighting=True,coverage_only=False):

    '''Compute the PseudoRange based on subregions and observations.'''

    iD_occupied_subregions = np.unique(SubRegions[Obs==1]) # iD of occupied subregions
    #####################
    if weighting==False: # in case if we just want to consider all subregions with at least one oservation in it
        pseudo_range=np.zeros_like(SubRegions)
        for iD in iD_occupied_subregions:
            mask_subregion = SubRegions == iD
            pseudo_range[mask_subregion]=1
        return pseudo_range.astype("float64")
    
    if coverage_only==True:
        t0=Time.time()
        pseudo_range=np.zeros_like(SubRegions)
        iteration=0
        for iD in iD_occupied_subregions:
            iteration+=1
            if Time.time()-t0>5:
                print("iD number = ",iD, "subregion ",iteration,"/",len(iD_occupied_subregions),end=" ")
                t0=Time.time()
            mask_subregion = SubRegions == iD
            n_obs = np.nansum(Obs[mask_subregion]) # total obs of the species in that subregion
            obs_area = np.sum(mask_subregion) / n_obs # individual area attrtributed to each observation point for coverage computation
            K=circkernel(obs_area) # circular kernel with this area
            covmap = Obs*mask_subregion # we only keep observation inside the specified subregion
            covmap = applykernel_pr(covmap,K) # we apply the circular kernel on each point
            covmap = covmap>0# we keep only the area hitted by the points
            coverage= covmap.sum() / (obs_area*n_obs) # proportion of the subregion hitted by
            pseudo_range[mask_subregion]= coverage
        #keep only places >0 to coompute Totsu
        maskposit = pseudo_range>0
        # add .astype() to prevent problem later (can therefore be considered as an int16 if not enforcing float64)
        Totsu= (threshold_otsu(pseudo_range[maskposit].flatten())).astype("float64")
        # create a plateau effect
        print("T_otsu = ",Totsu)
        pseudo_range[pseudo_range>Totsu]=Totsu
        # avoid problem here in divide by enforcing float to Totsu
        pseudo_range = pseudo_range.astype("float64")
        pseudo_range /= np.nanmax(pseudo_range) 
        return pseudo_range
            
    ######################
    totobs_sp = np.nansum(Obs) # total obs sp
    Total_area_subregions =0
    totobs_taxa = 0
    iteration=0
    print("Computing statistics on Obstaxa, please wait....")
    t0=0
    for iD in iD_occupied_subregions:
        iteration+=1
        if Time.time()-t0>5:
            print("iD number = ",iD, "subregion ",iteration,"/",len(iD_occupied_subregions),end=" ")
            t0=Time.time()
        mask_subregion = SubRegions ==iD
        Total_area_subregions += np.sum(mask_subregion) # summing over all occupied subregion to get total area of occupied subregions
        totobs_taxa += np.nansum(Obstaxa*mask_subregion) # all obs for the taxa in these subregions
    pseudo_range = np.zeros_like(SubRegions)
    print("Total species obs = ", totobs_sp)
    print("Total taxa obs =", totobs_taxa)
    #
    iteration = 0
    print("Computing Pseudo Range...")
    for iD in iD_occupied_subregions:
        iteration +=1
        if Time.time()-t0>5:
            print("iD number = ",iD, "subregion ",iteration,"/",len(iD_occupied_subregions),end=" ")
            t0=Time.time()
        mask_subregion = SubRegions == iD
        n_obs = np.nansum(Obs[mask_subregion]) # total obs of the species in that subregion
        n_obs_th = ( np.nansum(Obstaxa*mask_subregion) / totobs_taxa)*totobs_sp # theoretic number of obs in that subregion
        if n_obs_th != 0:
            obs_anomaly = n_obs/n_obs_th # observation anomaly
            obs_area = np.sum(mask_subregion) / n_obs # individual area attrtributed to each observation point for coverage coputation
            K=circkernel(obs_area) # circular kernel with this area
            covmap = Obs*mask_subregion # we only keep observation inside the specified subregion
            covmap = applykernel_pr(covmap,K) # we apply the circular kernel on each point
            covmap = covmap>0# we keep only the area hitted by the points
            coverage= covmap.sum() / (obs_area*n_obs)
            pseudo_range[mask_subregion]= coverage*obs_anomaly
        else:
            pass
    
    #keep only places >0 to coompute Totsu
    maskposit = pseudo_range>0
    # add .astype() to prevent problem later (can therefore be considered as an int16 if not enforcing float64)
    Totsu= (threshold_otsu(pseudo_range[maskposit].flatten())).astype("float64")
    # create a plateau effect
    print("T_otsu = ",Totsu)
    pseudo_range[pseudo_range>Totsu]=Totsu
    
    # avoid problem here in divide by enforcing float to Totsu
    pseudo_range = pseudo_range.astype("float64")
    pseudo_range /= np.nanmax(pseudo_range) 

    if plot==True:
        plt.figure(figsize=(10,10))
        plt.title("Reconstructed PseudoRange based on Observations")
        plt.imshow(pseudo_range)
        plt.colorbar(shrink=0.6)
        plt.grid(linestyle="--",color="grey",linewidth=0.2)
        plt.show(block=False)
    
    return pseudo_range


def Sm_Iucn(Iucn,nanmap = None,sig=50):

    '''Smooth the IUCN map using a Gaussian kernel and apply a distance-based weighting.'''
    
    dilated_nanmap = binary_dilation(nanmap, structure=np.ones((10,10))) # used to remove borders btw seas and land
    # Calculer les gradients pour repérer les jonctions entre régions
    gradient_x = sobel(Iucn, axis=1)  # Gradient horizontal
    gradient_y = sobel(Iucn, axis=0)  # Gradient vertical
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude[dilated_nanmap]=0
    frontiers = gradient_magnitude>0
    distances = distance_transform_edt(frontiers == 0)
    distances[nanmap]=None
    W = np.exp(-1/sig*distances)
    Iucn_smooth = fftconvolve(Iucn,gkern(sig=sig),mode="same")
    return Iucn_smooth*W + (1-W)*Iucn


def circ_kernel(size=50):

    '''Create a circular kernel of given size.'''

    S=2*size+1
    K=np.zeros((S,S))
    for i in range(S):
        for j in range(S):
            if ((i-size)**2+ (j-size)**2)**(1/2) <= size:
                K[i,j]= 1
    return K


def Obs_density(Map,plot_classic=False,until_1comp=False,by=1,tmax=100,NanMap=None,sigclassic=None):

    '''Compute the observation density map based on the input Map and parameters.'''

    dic_obs={} # dictionnary containing all points index for a given component
    continuous_regions=np.ones_like(Map)
    Map_nanremoved = Map.copy()
    if NanMap is not None:  
        Map_nanremoved[NanMap]=0
        continuous_regions[NanMap]=0
    iD_continuous_regions = label(continuous_regions, structure=np.ones((3, 3)))[0]
    obs_points = np.where(Map_nanremoved==1) # get obs positions
    dic_status={} # dictionnary of the form key:["alive",age]
    dead_comp =[] # list contatining all dying component during a given time step
    dead_comp_age=[] # list containing age of components
    dead_comp_time=[] # list containing birth time of living components during a given time step
    #
    Tmax=tmax
    #
    if until_1comp == True:  # the algorithm runs until only one component exists on the map
        #
        imax,jmax = np.shape(Map) # size of the map
        Tmax = max(imax,jmax) # run all possibles distances between points
    #####
    #
    print("tmax=",Tmax)
    #
    for t in np.arange(by,Tmax+by,by):
        #
        dic_ref = dic_obs # reference des composantes au temps (t-1)
        dic_obs={} # on vide dic_obs , va contenir les valeurs actualisées des composantes au temps (t)
        print("t=",t)
        effect_Map = applykernel(Map,circ_kernel(size=t))>=1
        if NanMap is not None:
            # remove oceans to avoid junctions between continents or islands
            effect_Map[NanMap]=0
        # compute iD for components
        iD_t=label(effect_Map, structure=np.ones((3, 3)))[0]
        # check which datapoint is in which component
        # create a list of sets like [grp1={1,2,3},...]
        nb_grp = iD_t.max()
        #
        if nb_grp == 1 or t==Tmax: # si on a atteint une seule composante ou qu'on a tester toutes les distances <Tmax on stoppe l'algorithme
            # kill all components
            #
            for key in list(dic_status.keys()):  # pour toute composante actuelle si l'algorithme se termine:
                #
                #
                if dic_status[key][0]=="alive":   # on termine toutes les composantes (on considère qu'elle sont mortes)
                    # kill it if it is alive
                    dead_comp.append(dic_ref[key]) # store dead component
                    dead_comp_age.append(dic_status[key][1]) # store age
                    dead_comp_time.append(dic_status[key][2])# store init time
                    dic_status[key][0]="dead"
            break # terminate everything
        ####
        # 
        for n in range(len(obs_points[0])):  # si l'algorithme tourne
            #
            #
            obs=[obs_points[0][n],obs_points[1][n]]
            (x,y)=obs
            grp_id = iD_t[x,y]
            #
            if grp_id in dic_obs : # si le groupe existe déjà dans le dictionnaire, on ajoute le point dans le set
                #
                dic_obs[grp_id].append(n)
            else:
                #
                dic_obs[grp_id]=[n] # we create this key  for the dictionnary and we associate obs to it
        #########
        # dic_ref
        emerging_comp=set()
        dying_comp =set()
        #
        keys_obs = list(dic_obs.keys())
        keys_ref = list(dic_ref.keys())
        #
        #
        for key in keys_ref:
            # does the key still exists?
            #
            if key in keys_obs: # the key still exists
                # did the value of the key have changed?
                #
                if dic_obs[key]==dic_ref[key]: 
                    #
                    # the value of the key hasn't changed
                    dic_status[key][0]= "alive"
                    dic_status[key][1]+=1 # age is increasing
                else:
                    #
                    # the value of the key has changed
                    #print("the value of the key has changed :")
                    #print("red",dic_ref[key])
                    #print("obs",dic_obs[key])
                    dic_status[key][0]="alive"
                    dead_comp.append(dic_ref[key]) # store dead component
                    dead_comp_age.append(dic_status[key][1]) # store age
                    dead_comp_time.append(dic_status[key][2]) # store init time
                    dic_status[key][1]=0 # reset age cause the comp changed
                    dic_status[key][2]=t # reset init time cause comp changed
                    #
            #########
            #
            else:
                # the key has been removed
                # so this key died (we set this key status to dead)
                dic_status[key][0] = "dead"
                dead_comp.append(dic_ref[key]) # store dead component
                dead_comp_age.append(dic_status[key][1]) # store age
                dead_comp_time.append(dic_status[key][2]) # store init time
                #
        #########
        #
        for key in keys_obs:
            #
            if key not in keys_ref: # the key appeared
                #
                #if a key apparead for the first time we need to
                #create a key status for it:
                dic_status[key]=["alive",0,t]
                #
    #############
    print("###################################")
    print("1.Construct component history: DONE")
    print("###################################")
    # for each group create a set with all points with a given 
    # we need to select largest lifespan component first, and then simplify if a component is inside one of the other
    # we can start from low values, then apply the kernel with the central value given by the age achieved by the component.
    unique_comps = np.unique(dead_comp)
    agelist=[]
    timelist=[]
    complist=[]
    for component in unique_comps: # pour chaque composante
        age=0
        times=[]
        complist.append(component)
        for n in range(len(dead_comp)):
            if dead_comp[n]==component:
                times .append( dead_comp_time[n])
                age+= dead_comp_age[n]
        agelist.append(age)
        timelist.append(min(times)) # moment de naissance de la composante (on prends ici le min car la même composante peut)
        # parfois réapparaître sous différents iD lorsque d'autres composantes dispraissent, il est donc nécéssaire d'éffectuer
        # un "matching" pour retrouver l'histoire complète d'une composante.
        #
    #####
    print("###################################")
    print("2.Tracking component history through time: DONE")
    print("###################################")
    # 1. prendre la composante la plus agée
    # 2. appliquer à tous les points à l'intérieur le kernel correspondant (ne pas appliquer sur les points déjà faits)
    # 3. stocker les points déjà utilisés
    #
    #
    newMap= np.zeros_like(Map)
    already_seen_points=[] # a list that keep track of the points already considered
    agelist_sorted = agelist.copy()
    agelist_sorted.sort()
    siglist=[]
    #
    while len(already_seen_points)<len(obs_points[0]):
        #
        if len(complist)==0:
            #
            break
            #
        #####
        print("Searching for remaining points in the left components...")
        #
        # find the points to consider
        selected_age = agelist_sorted[-1] # select the maximum age of all components
        idx_selected_age = agelist.index(selected_age) # get the index of that maximum
        selected_comp = complist[idx_selected_age] # get the component with that age
        #
        n=0 # iteration inside the component
        t_ini = Time.time()
        first_point=True
        
        
        for idx_selected_point in selected_comp:  # for a given component (begin with larger ones)
            n+=1
            if Time.time()-t_ini > 5 :  
                print("applying kernel for the selected component.... progress: ",round(n/len(selected_comp)*100,2),"%")
                t_ini = Time.time()
            #
            if idx_selected_point not in already_seen_points: # for points in that component, 
                #if their are not already been seen, apply the associated kernel size on these points
                x = obs_points[0][idx_selected_point] ; y=obs_points[1][idx_selected_point]
                if first_point == True:
                    accessible_region = iD_continuous_regions==iD_continuous_regions[x,y]
                    print("Computation of the accessible region for this component: DONE")
                    Map_comp = np.zeros_like(newMap) # create an empty map to apply kernels for this component
                    sig_param = timelist[idx_selected_age]
                    Nc = len(selected_comp)
                    GaussianKernel = gkern(sig=sig_param)*(sig_param)*(Nc>3) # probablement ça qui est lent!!!! 
                    print("Applying adaptative kernels ....")
                    print("#######################################################")
                    print("component = ",selected_comp)
                    print("sig_param = ",sig_param)
                    print("#######################################################")
                    first_point=False
                Map_comp[x,y]=1
                siglist+= [sig_param]
                already_seen_points.append(idx_selected_point)
                #print("already seen points =",already_seen_points)
                #
        #########
        del(agelist_sorted[-1]) # remove this value from the list of values
        del(agelist[idx_selected_age]) # remove this age in agelist
        del(complist[idx_selected_age]) # remove this component
        del(timelist[idx_selected_age]) # remove this time in the list
        
        if first_point == False:
            density_to_add = applykernel(Map_comp,GaussianKernel)*accessible_region # apply kernel
            print("Kernel with sig=",sig_param," applied for this component")
            sig_tresh = 1/(sig_param**2)
            density_to_add[density_to_add>10*sig_tresh]=sig_tresh + np.log(1+density_to_add[density_to_add>10*sig_tresh]-sig_tresh)
            newMap+=density_to_add # apply kernel
            print("number of remaining component:",len(complist))
            print("number of remaining points:", len(obs_points[0])- len(already_seen_points))
        #
    #####
    post_thrshld = threshold_otsu(newMap[obs_points])
    print("chosen threshold =",post_thrshld)
    print(" old max= ", np.nanmax(newMap))
    newMap[newMap>post_thrshld]=post_thrshld
    
    if NanMap is not None:
        print("Cropping nanmap...")
        newMap[NanMap]=None
    print("Plotting obs density with adaptative bandwidth....")
    
    return newMap






def EStar(Obs,RefRange,IucnDistSmooth,plot=True,NanMap=None,W_density=1.5,W_E=1,sig=None,ByComp=True,tmax=200,by=10,Refining_RefRange_with_Obs = False,WeightMap=None,KDE_mode="ClassicKDE",Ksig=None,densitymap=None):
    
    '''Compute the E* map based on observations, reference range, and IUCN distance smoothing.'''
    
    if KDE_mode not in ["fastKDE","ClassicKDE","ClassicKDE + Declustering","cosgkern + x/(1+x)"]:
        raise ValueError("/!\ KDE_mode should be a mode chosen in the following: fastKDE,ClassicKDE,ClassicKDE + Declusteing or cosgkern + x/(1+x) " )
    
    import numpy as np
    global IUCNalone
    IUCNalone=False
    # check if Obs is already in np.array format
    if Obs is not None:
        if isinstance(Obs, np.ndarray)==False:
            Map=np.array(Obs)
        else:
            Map=Obs
        #remove all None values 
        Map[np.isnan(Map)]=0
        # remove all number of observation greater than 1 in a single pixel
        Map[Map>1]=1
        N=Map.sum()
        nrow,ncol=np.shape(Map)
        print(N," observations in total")
    else:
        N=0
        nrow,ncol=np.shape(RefRange)
    if N<=10 and RefRange is not None:
        print("Very low number of observations (N=",N,"), IUCN range alone is considered")
        IUCNalone=True
    if N<30 and RefRange is None:
        print("Very low number of observations (N=",N,")/!\ result could be non reliable /!\ ") 
    if RefRange is not None and Obs is not None:
        if np.nansum(Obs*(RefRange>0))/N<0.4:
            print("More than 60% of data lie outside IUCN range extent, data is unclear... IUCN alone is considered")
            IUCNalone=True
        
    
    #This part follows instruction from fastKDE method 
    if IUCNalone==False:  
        if sig is None or KDE_mode=="fastKDE":
            print("mode_KDE = fastKDE")
            print("Beginning Kernel Desnity Estimation using fastKDE")
            print("Computing Fourier Transform of Observation Data...")
            ECF=np.fft.ifft2(Map)
            FObs = np.fft.fftshift(ECF)
            plt.figure()
            Magn=np.abs(FObs)
            Magn=Magn/Magn[nrow//2,ncol//2]
            Magn=Magn**2
            plt.figure(figsize=(5,5))
            Plot=Magn[nrow//2 - 100  : nrow//2 + 100, ncol//2 - 100: ncol//2 + 100]
            ##################################PLOT####################################
            if plot==True:
                plt.imshow(Plot)
                plt.title("F(Obs)")
                plt.xlabel("t1")
                plt.ylabel("t2")
                plt.colorbar(shrink=0.8)
                plt.contour(Plot,colors="white",linewidths=0.5)
                #plt.show(block=False)
            ##########################
            print("Filtering F(Obs) and extracting larger contiguous hypervolume such that |F(Obs)|² > 4(N-1)/N²")
            MaskMagn=(Magn>=4*(N-1)/(N**2))
            iD=label(MaskMagn, structure=np.ones((3, 3)))[0]
            MaskPhi0=iD==iD[nrow//2,ncol//2]
            ##########################################PLOT
            if plot==True:
                plt.figure(figsize=(5,5))
                plt.title("Mask for the Dumping Function Phi")
                plt.xlabel("t1")
                plt.ylabel("t2")
                plt.imshow(MaskPhi0[nrow//2 - 200 : nrow//2 + 200, ncol//2 - 200: ncol//2 + 200])
                plt.colorbar(shrink=0.8)
                #plt.show(block=False)
            ##################
            Phisq=Magn*MaskPhi0
            Phisq[Phisq==0]=None
            #########################################PLOT
            if plot==True:
                plt.figure(figsize=(5,5))
                plt.imshow(Phisq[nrow//2 - 200 : nrow//2 + 200, ncol//2 - 200: ncol//2 + 200])
                #plt.show(block=False)
            #########################
            print("Computing Dumping Function...")
            psy= N/(2*(N-1))*(1+(1-4*(N-1)/(N**2*Phisq))**(1/2))
            psy[np.isnan(psy)]=0
            ###########################################PLOT
            if plot==True:
                plt.figure(figsize=(5,5))
                plt.imshow(psy[nrow//2 - 200 : nrow//2 + 200, ncol//2 - 200: ncol//2 + 200])
                plt.title("Dumping Function psy")
                plt.xlabel("t1")
                plt.ylabel("t2")
                plt.colorbar()
                plt.contour(psy[nrow//2 - 200 : nrow//2 + 200, ncol//2 - 200: ncol//2 + 200])
                #plt.show(block=False)
            #######################
            K = np.fft.fft2(psy)
            K = np.fft.fftshift(K)
            K=np.abs(K)
            ####################################PLOT
            if plot==True:
                plt.figure(figsize=(5,5))
                plt.title("Optimal kernel after inverse Fourier")
                plt.imshow(K[nrow//2 - 400 : nrow//2 + 401, ncol//2 - 400: ncol//2 + 401])
                plt.colorbar()
                plt.contour(K[nrow//2 - 400 : nrow//2 + 401, ncol//2 - 400: ncol//2 + 401],colors="white",linewidths=0.5)
                #plt.show(block=False)
            ########################
            K=K[nrow//2 - 400 : nrow//2 + 401, ncol//2 - 400: ncol//2 + 401]
            Kmin=K.min()
            nk,ck=np.shape(K)
            Kmax=K.max()
            print(Kmax)
            K[K==Kmax]=0
            #####
            Mapdist=K<threshold_otsu(K)
            distance_transform = distance_transform_edt(Mapdist)
            distance_transform[distance_transform==0]=1
            inverse_weight=1/distance_transform
            K=K*inverse_weight
            #####
            #K[K<threshold_otsu(K)]=0
            K=K/K.max()
            E = applykernel(Map,K,NanMap=NanMap,WeightMap=WeightMap)
            
            
            
        else:
            if sig is not None and (KDE_mode=="ClassicKDE" or KDE_mode=="ClassicKDE + Declustering") :
                K=gaussian_kernel(sig)
            if Ksig is not None:
                print("specific kernel provided...")
                K=Ksig
            #######################################PLOT
#         if plot==True and (ByComp==False):
#             plt.figure(figsize=(5,5))
#             plt.title("Holey Kernel")
#             plt.imshow(K,cmap="viridis")
#             plt.colorbar()
#             plt.contour(K,colors="white",linewidths=0.5)
#             #plt.show(block=False)
        ########################
        ##
            if sig is not None and KDE_mode=="cosgkern + x/(1+x)":
                K = cosgkern(sig=sig)
                E=applykernel_pr(Map,K)
                logobs=E
                E[E<0]=0
                E=E/(1+E) # normalisation interessante
                #
        print("Observations density map")
        T0=Time.time()
        if KDE_mode=="ClassicKDE" or KDE_mode=="ClassicKDE + Declustering": 
            if densitymap is not None:
                E=densitymap
            else:
                E=applykernel(Map,K,NanMap=NanMap,WeightMap=WeightMap)
            #maxk= np.nanmax(K)
            #E= (E <=maxk*2)*E + (E>maxk*2)*(maxk*2 + np.log(1+ (E - maxk*2)*(E-maxk*2 >=0 ))) # to avoid over representation of some places
        elif ByComp == True or KDE_mode=="ByComponentKDE":
            E=Obs_density(Obs,tmax=tmax,by=by,NanMap=NanMap,plot_classic=False)
            
        print((Time.time()-T0)/60,"mn")
        
        
    ##
    if IUCNalone==True:
        E=np.zeros(np.shape(RefRange))
        logobs = np.zeros_like(RefRange)
              
    
#     if RefRange is not None:
#         E[np.isnan(RefRange)]=None
#     #
    
    ##################################PLOT
    if IUCNalone==False:
        
        if WeightMap is None:         # juste afficher le plot de densité
            if plot==True:
                plt.figure(figsize=(5,5))
                plt.title("Obs density")
                plt.imshow(E,cmap='viridis')
                plt.colorbar(shrink=0.8)
                plt.contour(E,linewidths=0.5,colors="white",levels=5)
                plt.xlabel("X(km)")
                plt.ylabel("Y(km)")
            logobs=E
            
        if WeightMap is not None:   # afficher le plot de densité basique et le resultat du cell declustering
            #
            
            if plot==True:
            
                E0 = applykernel(Map,K,NanMap=NanMap,WeightMap=None)

                fig, axs = plt.subplots(1,2,figsize=(10,10))  # Creates a 2x2 grid of subplots

                axs[0].set_title("Obs density")
                axs[0].imshow(E0,cmap='viridis')
    #             axs[0].set_xticks([])  # Remove x-axis ticks
    #             axs[0].set_yticks([])  # Remove y-axis ticks
                axs[0].contour(E0,linewidths=0.4,colors="white",levels=5)
                axs[0].set_xlabel("X(km)")
                axs[0].set_ylabel("Y(km)")
                axs[0].grid(True,color="grey",linewidth="0.5",alpha=0.5)

                ######

                axs[1].set_title("Obs density after Cell Declustering")
                im2=axs[1].imshow(E,cmap='viridis')
    #             axs[1].set_xticks([])  # Remove x-axis ticks
    #             axs[1].set_yticks([])  # Remove y-axis ticks
                axs[1].contour(E,linewidths=0.4,colors="white",levels=5)
    #             axs[1].set_xlabel("X")
    #             axs[1].set_ylabel("Y")
                axs[1].grid(True,color="grey",linewidth="0.5",alpha=0.5)
            #
            logobs=E
    #####################
        ##
    if RefRange is not None: # RefRange exists
        #
        if Refining_RefRange_with_Obs == True and IUCNalone==False:
            #
            print("Reffining Reference range input with observation density")
            weights = np.unique(RefRange) # all weights of the reference range
            for weight in weights:
                region = RefRange==weight
                dens_values_in_region = logobs[region]
                mean_value= np.nanmean(dens_values_in_region)
                correcting_coeff = weight / mean_value
                E[region]=E[region]*correcting_coeff # change mean in that region but keep the shape density
            Rep = Sm_Iucn(E,nanmap = NanMap,sig=IucnDistSmooth)
            Rep[NanMap]=None
            Rep/=np.nanmax(Rep) # normalisation
            return Rep,logobs
                
        if Refining_RefRange_with_Obs == False or IUCNalone==True:
            #
            #####
            
            I=RefRange
            I0 = RefRange==0
            print("Computing Reference range sides...")
            I[np.isnan(I)]=0
            Sides = Sm_Iucn(I,NanMap,sig=IucnDistSmooth)
            
            if IUCNalone==True:
                Rep = Sides*W_E
                Rep[Rep>1]=1
                if NanMap is not None:
                    Rep[NanMap]=None
                return Sides,logobs
            print("Combine density and background E, with a linear combination")
            if densitymap is not None:
                print("specific density map provided...")
                E=densitymap
            E/=np.nanmax(E)
            Rep=(E*W_density + Sides*W_E) # combining different pieces of informations
#             plt.figure()
#             denshist = (E*W_density).flatten()
#             denshist=denshist[np.isnan(denshist)==0]
#             Ehist = (Sides*W_E).flatten()
#             Ehist = Ehist[np.isnan(Ehist)==0]
#             plt.hist(denshist,bins=50,alpha=0.5)
#             plt.hist(Ehist,bins=50,alpha=0.5)
#             #plt.show(block=False)
            Rep[Rep>1]=1 # 
            if NanMap is not None:
                Rep[NanMap]=None
            Rep/=np.nanmax(Rep) # normalization
            return Rep,logobs
    
    if RefRange is None:
        Rep = logobs.copy()
        Rep /= np.nanmax(Rep)
    return Rep,logobs

###### 

def CurrRange(Obs,RefRange,HS,IucnDistSmooth,plot=False,sizecoeff=10,NanMap=None,plotFourier=False,W_density=1,W_E=0.5,sig=10,tmax=200,by=10,ByComp=False,Refining_RefRange_with_Obs = False,mode="Permanence of ratios", Declustering = True,by_declust=5,KDE_mode="ClassicKDE + Declustering",Ksig=None,densitymap=None):
    
    """

    CurrRange(Obs,RefRange,HS,IucnDistSmooth,plot=False,sizecoeff=10,NanMap=None,plotFourier=False,W_density=1,W_E=0.5,sig=10,tmax=200,by=10,ByComp=False,Refining_RefRange_with_Obs = False,mode="Permanence of ratios", Declustering = True,by_declust=5,KDE_mode="ClassicKDE + Declustering")

    CurrRange is a function that compute an estimated realised range of a species based on Environemental indicators in the form of an Habitat Suitability map, and spatial constraints in a form of a reference range or a reconstructed range extent. CurrRange uses Permanence of Ratios to incorporate both information together.

    Args:
        HS (2D numpy array, float64) (mandatory): A 2D numpy array. Habitat Suitability map, or equivalently all estimation of P(site being part of the realised range | Env). For best results, make sure the maps are output from models that used a 50% presence prevalence in the training data, and spatial bias are accounted as much as possible. IMPORTANT: make sure the 2D array displays values between 0 and 1 and NaN regions are set to 0.

        Obs (2D numpy array, float64) (mandatory): A 2D numpy array. Occurences should be displaying the number of observation in the pixel, all pixels without occurences set at 0.

        sig (int) (default = 30): the standard deviation of the 2 dimensional gaussian kernel used to reconstruct an occurence density. sig constitute an important parameter, in a European scale, usually sig is around 30km but it can change your result. Be sure to check visually on simple examples to best choose this parameter. Adjust the parameter util occurences likely to be part of the same range continuum lie in the same patch. At European scale, around 30 km.

        RefRangeDistSmooth (int) (default = 50): standard deviation of the 2 dimensional gaussian kernel used to smooth the spatial constraint before Permanence of Ratios incorporation. It could be usefull in the case where if typerange = "PseudoRange", to smooth transition between computed subregions weighs. This will avoid abrupt transition in the output. At Euopean scale usually 50 km.

        WE (float): weighting coefficient given to the reference range part (E) compared to the reconstructed occurence density (dens) part. P(site part of the realised range | x,y) = I( WE*E + Wdens*dens), where I is a cutting function that sent all values greater than 1 to 1. WE and Wdens are real values. As an example, WE=0.7, Wdens=0.7, means that both E and dens should be high to reach the maximum confidence of 1. If the density is at its maximum and E=0 (the site is not part of the reference range), the maximum value attainable is 0.7. (for futher information see article)

        Wdens (float): weighting coefficient given to the reconstructed occurence density part compared to the reference range  part.

        by_declust (int): the step size (pixels), should be an integer, when searching the optimal space division for cell declustering. Lower value is more precise but slower, feel free to increase bydeclust if the CellDeclustering process is too long. Usuall values for European scale are around 30.

        plot (bool) (default=False): if True show all plots during the process


    Returns:
        2D array float 64, corresponding to the continuous map P(site is part of the realised range | x,y & Env)
    """
    
    
    global IUCNalone
    T_initial = Time.time()
    
    print("Initialisation....")
    print("Checking HS interval....")
    if isinstance(HS, np.ndarray)==False:
        raise ValueError("Error: Habitat Suitability Map is not in a form of a numpy 2D-array, make sure that Habitat Suitability map is in the suitable form, assure that Habitat Suitability is between 0 and 1, NaN values are accepted but will be replaced by default by 0")
    Hs=HS.copy()
    
    CellsdiffNone=(np.isnan(Hs)==False)*(Hs!=0)
    TotalNumberofCells=(CellsdiffNone).sum()
    X=np.linspace(0.001,0.999,100)
    if Hs.max()>1:
        print("Normalisation needed for HS")
        print("Normalising HS....")
        Hs[Hs==Hs.max()]=0
        Hs=Hs/10000
    
    if NanMap is not None:
        if isinstance(NanMap, np.ndarray)==False:
            raise ValueError("Error: Nan Map is not in a form of a numpy 2D-array, make sure that the NanMap is a binary boolean numpy 2D-map with ones where there are non referenced pixels such as seas for terrestrial species and zeros everywhere else" )
        
      ###################################PLOT
    if Obs is not None:
        if isinstance(Obs, np.ndarray)==False:
            raise ValueError("Error: Observation Map is not in a form of a numpy 2D-array, make sure that the observation map is in the suitable form, assure that all values are 0 except where observations occur, there, values should be ones. NaN values are accepted but will be replaced by default by 0, values > 1 will be replaced by default by 1")
        else:
            Map=Obs
        #remove all None values 
        Map[np.isnan(Map)]=0
        # remove all number of observation greater than 1 in a single pixel
        Map[Map>1]=1 # for plottin gpoints
        xobs,yobs=np.where(Map==1)
        #######################################
    else:
        IUCNalone=True
        Map=None
        xobs=[];yobs=[]
    ##############################################################################
    ####################################FIGURES
    if plot==True:
        plt.figure(figsize=(10,10))
        plt.imshow(Hs)
        plt.colorbar(shrink=0.6)
        plt.title("Habitat Suitability Map")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.scatter(yobs,xobs,s=5,c="red",marker="o",label="Observation points",alpha=0.1)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        plt.legend()
        
        #plt.show(block=False)
    
    if RefRange is not None:
        if isinstance(RefRange, np.ndarray)==False:
            raise ValueError("Error: Reference range is not in a form of a numpy 2D-array. NaN values are accepted, all possible weighting is accepted, make just sure that values are reals")
        else:
            Iucn_loc=RefRange
        
        if plot==True:
            plt.figure(figsize=(10,10))
            plt.imshow(Iucn_loc)
            plt.title("Reference Range")
            plt.xlabel("X(km)")
            plt.ylabel("Y(km)")
            plt.colorbar()
    else:
        Iucn_loc=None
    
    if KDE_mode == "ClassicKDE + Declustering" and densitymap is None:
        print("#################")
        print("Debiaising non uniform sampling by Cell Declustering...")
        print("#################")
        WeightMap = celld(Obs=Map,NanMap=NanMap,sig=sig,gridcellsize=10,plotgrid=True,mode="diagnose",listdivgrids=None,background=Hs,by_declust=by_declust,plot=plot,Ksig=Ksig)
    else:
        WeightMap = None
    Estar,logobs=EStar(Map,Iucn_loc,IucnDistSmooth,NanMap=NanMap,plot=plotFourier,W_density=W_density,W_E=W_E, ByComp = ByComp , sig=sig,tmax=tmax,by=by,Refining_RefRange_with_Obs =Refining_RefRange_with_Obs,WeightMap=WeightMap,KDE_mode=KDE_mode,Ksig=Ksig,densitymap=densitymap)
    if plot==True:
        plt.figure(figsize=(10,10))
        plt.title("Output from Estar function (Rep)")
        plt.imshow(Estar)
        plt.colorbar(shrink=0.6)
        level_incr=np.nanmax(Estar)/5
        plt.contour(Estar,linewidths=0.2,linestyles='--',colors="white",levels=[level_incr*k for k in range(1,6)])
        plt.grid(linestyle="--",linewidth=0.2,color="grey")
        #plt.show(block=False)
    if mode=="Bayes":
        if IUCNalone==True:
            plt.figure(figsize=(20,20))
            CR=Estar*HS
            plt.imshow(CR,cmap="viridis")
            plt.colorbar()
            plt.xlabel("X (km)")
            plt.ylabel("Y (km)")
            plt.title("Estimated Current Range IUCNsmoothXHS")
            plt.scatter(yobs,xobs,c="white",s=10,alpha=0.5)
            #plt.show(block=False)
            return CR,logobs,Estar
        ##############################################################################
        ####################################FIGURES
        if plot==True:
            plt.figure(figsize=(20,20))
            plt.imshow(Estar)
            plt.colorbar()
            plt.xlabel("X (km)")
            plt.ylabel("Y (km)")
            plt.title("Constraining map E*")
            plt.scatter(yobs,xobs,c="red",s=10,alpha=0.2)
            #plt.show(block=False)
        ###########################################
        ##############################################################################
        print("Computing P(E*s|Obs)...")
        WObs=Estar[Map==1]
        vector_1d = WObs.flatten()
        vector_1d = vector_1d[~np.isnan(vector_1d)]
        print(vector_1d)
        print(len(vector_1d))
        ###### compute the correct number of bins
        var=vector_1d.var()
        mu=vector_1d.mean()
        X=np.linspace(0.001,0.999,100)
        Y=[Beta2(x,mu,var) for x in X]

        print("Beta fit")
        plt.figure(figsize=(10,10))
        plt.hist(vector_1d,density=True)
        plt.plot(X,Y,'--',color="red",label="Beta fit")


        print("Correcting E*s|Obs/E*s desnity...")
        maxY=0 # max ini
        for yidx in range(len(Y)):
            if Y[yidx]>maxY:
                maxY=Y[yidx] # new value for max
            if Y[yidx]<maxY:
                Y[yidx]=maxY
        Y=np.array(Y)
        Y[Y>2]=2 # use to limit the range of differences between the lower weighted part and the heigher ones
        # avoid "cropp" effects on the map and alow for more smooth wieghting

        print("Beta fit")
        plt.plot(X,Y,'--',color="orange",label="Beta fit corrected for decreased")
        #plt.show(block=False)

        bin_edges,relative_freq_Es_knw_Obs = (X,Y[1:])
        print("bin_edges and relative_freq",bin_edges,relative_freq_Es_knw_Obs)

        M=np.where(Estar>0.01) # au dessus de 0 car après la diffusion en utilisant Fourrier, on a des résidus proches de 0
        # un peu partout ! On retire les "0" de l'histogramme pour se concentrer sur les zones en lien avec la distribution
        # et non les zones "vides". 
        vector_1dEs = Estar[M]

    #     vector_1dEs = vector_1dEs[~np.isnan(vector_1dEs)]
    #     bin_edges,relative_freq_Es = genhist(vector_1dEs)

        WeightMap= np.zeros_like(Estar)
        for k in range(len(bin_edges)-1):
            less_binmap = Estar <= bin_edges[k+1]
            more_binmap = bin_edges[k] < Estar
            betw_binmap = more_binmap*less_binmap
            WeightMap[betw_binmap]=relative_freq_Es_knw_Obs[k]

        WeightMap[Estar > bin_edges[k+1]]=np.nanmax(WeightMap)

        plt.figure()
        plt.title("Computed Weight Map")
        plt.imshow(WeightMap)
        plt.colorbar()
        #plt.show(block=False)
    #     WeightMap = p(Estar)-p0
        WeightMap[WeightMap<0]=0
        CR= HS*WeightMap
        T=Time.time()
        CR[NanMap]=None

        #NORMALISATIONNNN 
    #     if IUCN is not None:
    #         maxCR=np.nanpercentile(CR*Iucn_loc,97.5)
    #         if maxCR==0:
    #             print("Non satisfying normalization found, normalization performed using optimized Otsu treshold")
    #             NonNan=CR*Iucn_loc
    #             NonNan[np.isnan(NonNan)]=0
    #             maxCR=threshold_otsu(NonNan)
    #     else:
    #         maxCR=np.nanpercentile(CR,97.5)
    #         if maxCR==0:
    #             print("Non satisfying normalization found, normalization performed using optimized Otsu treshold")
    #             NonNan=CR.copy()
    #             NonNan[np.isnan(NonNan)]=0
    #             maxCR=threshold_otsu(NonNan)

    #     CR[CR>maxCR]=maxCR
    #     CR=CR/maxCR

        CR = CR/np.nanmax(CR)
          ##############################################################################
        ####################################FIGURES
        if plot==True:
            plt.figure(figsize=(20,20))
            plt.imshow(CR,cmap="viridis")
            plt.colorbar(shrink=0.7)
            plt.xlabel("X (km)")
            plt.ylabel("Y (km)")
            plt.title("Estimated Current Range P(Ps|E*s)")
            #plt.show(block=False)
        ###########################################
        ##############################################################################
        ########

        T_final=Time.time()

        print("Total computing time for the species",(T_final-T_initial)/60,"mn")
    
    if mode=="Permanence of ratios":
        if NanMap is not None:
            Hs[NanMap]=0
            Estar[NanMap]=0
        pa=0.5  # prior information on a random site with no inforation on it (0.5 is most uncertain state)
        pa_kn_hs = Hs.copy() # get Hs
        pa_kn_space = Estar.copy() # get spatial information
        pa_kn_hs = pa_kn_hs/ np.nanmax(pa_kn_hs) # normalise both
        pa_kn_space = pa_kn_space/ np.nanmax(pa_kn_space) # normalise both
        # due to noise when using the Fourrier convolution with IUCN, we need to remove very low values to avoid problems of computation
        nullPenv=HS<0.0001
        nullPxy=Estar<0.0001

        pa_kn_hs[nullPenv]=0.001 # just a tamporary value for computation and avoid 0 values and problems of division
        pa_kn_space[nullPxy]=0.001

        CR = ((1-pa)/pa) / ( (1-pa)/pa +   (1-pa_kn_hs)/pa_kn_hs * (1-pa_kn_space)/pa_kn_space   )
        # after computation enforce that if Penv=0 or Pxy=0 the result is 0
        CR[nullPenv]=0
        CR[nullPxy]=0
        #CR=CR/np.nanmax(CR)
        
        if NanMap is not None:
            CR[NanMap]=None

          ##############################################################################
        ####################################FIGURES
        if plot==True:
            plt.figure(figsize=(10,10))
            plt.imshow(CR,cmap="viridis")
            plt.colorbar(shrink=0.6)
            plt.xlabel("X (km)")
            plt.ylabel("Y (km)")
            plt.title("Estimated Current Range")
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
            #plt.show(block=False)
        ###########################################
        ##############################################################################
        ########
    
    return CR,logobs,Estar



def runoverfile(hsfolder,obsfolder,obstaxafile,sig=30,subregionfile=None,RefRangeDistSmooth=50,WE=0.7,Wdens=0.7,bydeclust=40,typerange = "PseudoRange",NaNvalue=None,savefigfolder=None,outputtype="CR",plot=False,Ksig=None,bynx=10,comp1percent=50,maxpoints=10000,HStreatment="nanmin",Reftreatment="nanmin",KDE_mode="ClassicKDE + Declustering",listnamesformat=[],listvalidHSnames=[],tr1=1/10,tr2=5/10):
    
    """

    runoverfile(hsfolder,obsfolder,obstaxafile,sig=30,subregionfile=None,RefRangeDistSmooth=50,WE=0.7,Wdens=0.7,bydeclust=40,typerange = "PseudoRange",NaNvalue=None,savefigfolder=None, outputtype="CR",plot=False)

    runoverfile is a function that run function CurrRange over an entire file, please make sure that the chosen files contain elements that correpsonds in order: i.e first element of hsfolder file corresponds to the first element in obsfolder etc... To do so, all file can begin with the name of the species, this permits to ordeer them alphabetically. For more information about CurrRange function type help(CurrRange)

    Args:
        hsfolder (str) (mandatory): the path of the folder containing the raster files (tiff or tif format) for the Habitat Suitability maps, or equivalently all estimation of P(site being part of the realised range | Env). For best results, make sure the maps are output from models that used a 50% presence prevalence in the training data, and spatial bias are accounted as much as possible. Make sure that the given path is given using backslashes "\" rather than "/". 

        obsfolder (str) (mandatory): the path of the folder containing the observation raster files (tif or tiff format). For each raster, occurences should be displaying the number of observation in the pixel, all other pixels set at 0.

        obstaxafile (str) (Only if typerange = "PseudoRange"): the path to the file (tif or tiff format) corresponding to a unique raster following the same principle as files in obsfolder. obstaxafile corresponds to all occurences of a reference taxa. This input serves the purpose of accounting to non uniform Sampling Effort over your study area.

        sig (int) (default = 30): the standard deviation of the 2 dimensional gaussian kernel used to reconstruct an occurence density. sig constitute an important parameter, in a European scale, usually sig is around 30km but it can change your result. Be sure to check visually on simple examples to best choose this parameter. Adjust the parameter util occurences likely to be part of the same range continuum lie in the same patch. At European scale, around 30 km.

        subregionfile (str) (if rangetype = "PseudoRange"): a raster file (tif or tiff) with partition of the study area into meaningful divisions for the species (if an occurence is inside a subregion, the species is likely to be part of the whole subregion). Values corresponds to arbitrary iD in a form of integer values.

        RefRangeDistSmooth (int) (default = 50): standard deviation of the 2 dimensional gaussian kernel used to smooth the spatial constraint before Permanence of Ratios incorporation. It could be usefull in the case where if typerange = "PseudoRange", to smooth transition between computed subregions weighs. This will avoid abrupt transition in the output. At Euopean scale usually 50 km.

        WE (float) = weighting coefficient given to the reference range part (E) compared to the reconstructed occurence density (dens) part. P(site part of the realised range | x,y) = I( WE*E + Wdens*dens), where I is a cutting function that sent all values greater than 1 to 1. WE and Wdens are real values. As an example, WE=0.7, Wdens=0.7, means that both E and dens should be high to reach the maximum confidence of 1. If the density is at its maximum and E=0 (the site is not part of the reference range), the maximum value attainable is 0.7. (for futher information see article)

        Wdens (float) = weighting coefficient given to the reconstructed occurence density part compared to the reference range  part.

        bydeclust (int) = the step size (pixels), should be an integer, when searching the optimal space division for cell declustering. Lower value is more precise but slower, feel free to increase bydeclust if the CellDeclustering process is too long. Usuall values for European scale are around 30.

        typerange (str) (default = "PseudoRange") : the method used for reference range reconstruction,
        > "PseudoRange" : the reconstruction use subregion selections based on sampling effort (need obstaxafile and subregionfile).
        > "PseudoRange covonly" : if some sites are oversampled and leads to an overflow of presences by absences.
        > "PseudoRange presonly" : select the entire subregion if at least one occurence is in the subregion.
        > "OBR": Habitat Suitability is binarised into suitable and unsuitable habitats, using Otsu thresholding (all values > otsu threshold  are considered suitble patches). All suitable patches containing occurences are noted S1. All patches of suitable habitats under a distance threshold T but with no occurences (S0 pathces) are selected alongside S1 patches to form the reference range.  For "OBR" T corresponds to the maximum of all minimum distances between pairs of occurences. 
        >"LR": same principle as "OBR" but T is chosen to be the first quartile of minimal edge to edge distances between S1 and S0 patches.
        >"MCP: Habitat Suitability is binarised into suitable and unsuitable habitat using Otsu thresholding. In each suitable patch containing more than 3 occurences, a minimum convex poygon is produced using extremal occurences as summit. The union of all polygons constitue the reference range.

        savefigfolder (str) (mandatory): the path of the folder where you want the output tiff to be produced.

        outputtype (str) (default= "CR"): the output type desired.
        > "CR": the continuous raster P( site is part of the realise range | Env & x,y)
        > "Binary Boyce": a trinary output (values of 0 , 0.5 and 1), using method from Hizel et al
        > "Binary Otsu": a binary output using Otsu thresholding based on predicted values at occurences points.
        > "Cut50": a binary output with a threshold of 0.5.
        >"CR + BinaryBoyce" both continuous and binary maps using Boyce Index are computed
        >"CR + Cut50" both continuous and binary maps using the threshold 0.5 are computed

        plot (bool) (default =False): if plot is True, plot everything for a diagnosis of th entire process.

        listnamesformat (list), [Obsformat,HSformat] , example: if all files in the Observation folder are in the form Obs_speciesname_1kmresolution.tif and HS_speciesname_1kmresolution.tif for files in HS folder, with speciesname is the variable part the user need to specify in listnamesformat ["Obs_XxX_1kmresolution.tif","HS_XxX_1kmresolution.tif"] using XxX to indicate the variable part that the function will search to align files.

        maxpoints (int), number of occurences point to be sampled from the total observation map to compute a network if KDE_mode = "network KDE"

        listvalidHSnames (list of str), giving the names of authorized HS file to be computed, for example if the user want to run runoverfile but only on a subpart of the entire file, the user need to specify which HSfiles should be used


    Returns:
        tif rasters and png plots (if plot is True) in the savefigfolder specified
    """
    if savefigfolder is not None:
        if os.path.exists(savefigfolder + "/continuous")==False:
            os.makedirs(savefigfolder + "/continuous" )
        if os.path.exists(savefigfolder + "/binary")==False:
            os.makedirs(savefigfolder + "/binary" )
        if os.path.exists(savefigfolder + "/plots")==False:
            os.makedirs(savefigfolder + "/plots" )
        if os.path.exists(savefigfolder + "/spatial extent")==False:
            os.makedirs(savefigfolder + "/spatial extent" )


    # extract Obstaxa as reference if typerange is PseudoRange
    if typerange == "PseudoRange":
        Obstaxa=np.array(Image.open(obstaxafile))
        print("obstaxa extracted")
    
    idxfolder=0

    #####################################################################
    # FOR ALL FILES
    #####################################################################
    #for obsfile,hsfile in zip(os.listdir(obsfolder),os.listdir(hsfolder)):
    #####################################################################
    # FOR BIRDS DATA
    ####################################################################
    # we only take birds that are part of the STOC data (not all European Birds!)

    #if birdmode==True:
    #    path_birdObsSTOC = "C:/Users/hoare/OneDrive/Bureau/SAUVEGARDE 9/ENS/ENS Lyon (wd)/Cours/Cours/MASTER/SEMESTRE 3/LECA GRENOBLE/BirdsMaxime/STOCdata/AllBirds"
    #    listvalidSTOC = [birdfile[:-9].replace("_"," ") for birdfile in os.listdir(path_birdObsSTOC)]
    #else:
    #    listvalidSTOC=[]

    if os.path.exists(typerange): # if a path is provided for typerange
        listidxobs,listidxhs,listidxrefrange = allign2([obsfolder,hsfolder,typerange],listnamesformat=listnamesformat)
        refrangenames = os.listdir(typerange) # names of the files of reference range
    else:
        listidxobs,listidxhs = allign2([obsfolder,hsfolder],listnamesformat =listnamesformat)

    hsnames=os.listdir(hsfolder)
    obsnames=os.listdir(obsfolder)

    for k in range(len(listidxobs)): # get all files corresponding to the species, Obs HS and RefRange when typerange is specified as a path
        idxobs = listidxobs[k] ; idxhs=listidxhs[k]
        obsfile = obsnames[idxobs]
        hsfile = hsnames[idxhs]
        if os.path.exists(typerange):
            idxrefrange = listidxrefrange[k] 
            RefRange = np.array(Image.open(typerange + "/"+ refrangenames[idxrefrange])) # extract data here !!! 
            if Reftreatment=="nanmax":
                RefRange[RefRange==np.nanmax(RefRange)]=0
            if Reftreatment=="nanmin":
                RefRange[RefRange==np.nanmin(RefRange)]=0
            RefRange=RefRange.astype("float64")
            RefRange/=np.nanmax(RefRange)

        #if os.path.exists(typerange): # if a path is provided, go inside it to find the reference range
        #        print("RefRange folder detected")
        #        folderref = typerange
        #        listiucnnames=os.listdir(folderref)
        #        idxiucn=listiucnnames.index(spname+".tif")
        #        filename=listiucnnames[idxiucn]
        #        RefRange=np.array(Image.open(folderref + "/" + filename))
        #        RefRange[RefRange==np.min(RefRange)]=0
        #        RefRange=RefRange/np.max(RefRange)
        #        print("corresponding IUCN map =",filename)
        #else:
        #    print("no file named ",typerange," find")
            
        #####################################################################

        plt.close('all') # close all figs

        
        if listvalidHSnames==[]:
            listvalidHSnames = os.listdir(hsfolder)
        if os.path.exists(savefigfolder+"/continuous/"+obsfile)==False and hsfile in listvalidHSnames:

            try:
                HS = np.array(Image.open(hsfolder +"/" + hsfile))
                Obs = np.array(Image.open(obsfolder + "/" + obsfile))
                print("test corespondance")
                print("Obsfile=",obsfile)
                print("HSfile=",hsfile)
                if os.path.exists(typerange):
                    print("RefRangefile=",refrangenames[idxrefrange])
                if np.nansum(Obs<0)!=0:
                    print("negative values detected, replaced with 0")
                    Obs[Obs<0]=0
                print("checking Habitat Suitability format....")
                if HStreatment == "nanmin":
                    nanmap =HS==HS.min()   ############################### CHANGE 4 NANVALUE
                    mini=np.min(HS)
                    HS[HS==mini]=0
                    HS=HS.astype("float64")
                    HS/=np.nanmax(HS)
                if HStreatment == "nanmax":
                    nanmap =HS==HS.max()   ############################### CHANGE 4 NANVALUE
                    maxi=np.max(HS)
                    HS[HS==maxi]=0
                    HS=HS.astype("float64")
                    HS/=np.nanmax(HS)
                
                print("Computing range...")

                if typerange in ["PseudoRange","PseudoRange covonly","PseudoRange presonly"]: # we need the real points here, with the same bias of observations as known for the taxa of reference
                    SubRegions = np.array(Image.open(subregionfile))
                    SubRegions=SubRegions.astype("float64")
                    #SubRegions[nanmap]=0
                    if typerange == "PseudoRange":
                        RefRange = PseudoRange(SubRegions,Obs,Obstaxa,plot=plot,NanMap=nanmap) # add a NanMap parameter to specify nan regions
                    if typerange == "PseudoRange covonly":
                        RefRange = PseudoRange(SubRegions,Obs,plot=plot,NanMap=nanmap,coverage_only=True)
                    if typerange == "PseudoRange presonly":
                        RefRange = PseudoRange(SubRegions,Obs,plot=plot,NanMap=nanmap,weighting=False)

                ##### at this point we can reduce the number of points for faster computation and reducing bias in distribution details
                print("checking for point overload before network density estimation or range estimation methods...")

                npoints=np.nansum(Obs>=1)
                if npoints>maxpoints:
                    print("point overload, reducing number of points for network calculation")
                    nb_to_remove = npoints-maxpoints
                    x,y=np.where(Obs>=1)
                    idxpoints = range(len(x))
                    idxchosen = rd.sample(idxpoints,k=nb_to_remove)
                    xchosen = [x[i] for i in idxchosen]
                    ychosen = [y[i] for i in idxchosen]
                    reducedObs=Obs.copy()
                    reducedObs[xchosen,ychosen]=0 # we remove these points for faster computation and reducing bias
                    newnpoints = np.nansum(reducedObs>=1)
                    print("smaller data composed of ",newnpoints,' compared to initial ',npoints)
                else:
                    reducedObs=Obs
                    print("number of points are already small enough for fast computation")


                if typerange=="OBR":
                    print("Computing Observation Based Restriction Map....")
                    RefRange = OBRmap(reducedObs,HS)
                if typerange== "LR":
                    RefRange = LRmap(reducedObs,HS)
                if typerange=="MCP":
                    RefRange=MCP(reducedObs,HS)
                
                #if os.path.exists(typerange): # if a path is provided, go inside it to find the reference range
                #    folderref = typerange
                #    filename=os.listdir(folderref)[idxfolder]
                #    RefRange=np.array(Image.open(folderref + "/" + filename))
                #    idxfolder+=1
                
                # save the range (spatial extent)
                saveTIF(hsfolder +"/" + hsfile,RefRange,savefigfolder+"/spatial extent/"+obsfile[:-4]+".tif")

                if KDE_mode=="network KDE":
                    density = generate_density_map(Obs)
                if KDE_mode=="ClassicKDE + Declustering":
                    density= None

                Cr,logobs,estar = CurrRange(Obs=Obs,HS=HS,RefRange=RefRange,mode="Permanence of ratios",
                                            KDE_mode="ClassicKDE + Declustering",sig=sig,plot=plot,
                                            Refining_RefRange_with_Obs=False,IucnDistSmooth =RefRangeDistSmooth,
                                            NanMap = nanmap,W_E=WE, W_density=Wdens,
                                            by_declust=bydeclust,Ksig=Ksig,densitymap=density)
                        ######
                if outputtype=="CR" or outputtype=="CR + Binary Boyce" or outputtype=="CR + Cut50":
                    
                    plt.figure()
                    plt.title("$P(s \subset \Omega | Env \cap x,y)$")
                    plt.imshow(Cr,cmap="jet")
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.colorbar()
                    plt.grid(linestyle="--",linewidth=0.2,color="white")
                    if savefigfolder is not None:
                        plt.savefig(savefigfolder+"/plots/"+obsfile[:-4]+".png",dpi=200)
                        ##plt.show(block=False)              
                    print("Save TIF file for continuous map prediction P(s in Realised Range | Env, x,y)")
                    print("Values between 0 and 1: encoded with integers between 0 and 1000. Divide by 1000 to retrive initial values")
                    print("Convert output into integers map for efficient memory usage, NaN values are set to -1000")
                    intmap = Cr*1000
                    intmap[np.isnan(intmap)]=-1000
                    intmap = intmap.astype("int32")
                    saveTIF(hsfolder +"/" + hsfile,intmap,savefigfolder+"/continuous/"+obsfile[:-4]+".tif")
                    plt.close()
                
                if outputtype=="Binary Boyce" or outputtype=="CR + Binary Boyce":
                    print("Binarisation using Boyce index")
                    Crbin=BoyceIndexTresh(Cr,Obs,NanMap=nanmap,plot=plot,plotFourier=False,HSxIUCN=None,save=obsfile+"output.png",path=savefigfolder,thresh_default=0.5)
                    print("Values between 0 and 1: encoded with integers between 0 and 1000. Divide by 1000 to retrive initial values")
                    print("Convert output into integers map for efficient memory usage, NaN values are set to -1000")
                    binintmap = Crbin[0]*1000
                    binintmap[np.isnan(binintmap)]=-1000
                    binintmap = binintmap.astype("int32")
                    saveTIF(hsfolder +"/" + hsfile,binintmap,savefigfolder+"/binary/"+obsfile[:-4]+".tif")

                if outputtype=="Binary Otsu":
                    Crbin=Otsuthresh(Cr,Obs,NanMap=nanmap,savepath=savefigfolder+ "/" + obsfile+"binary_otsu.png",output="crbin")
                    saveTIF(hsfolder +"/" + hsfile,Crbin,savefigfolder+"/"+obsfile[:-4]+"binary_otsu.tif")

                if outputtype=="Cut50" or outputtype=="CR + Cut50":
                    Crbin=Cut(Cr,Obs,NanMap=nanmap,savepath=savefigfolder+ "/plots/"+ obsfile[:-4]+"_binary50.png",output="crbin",tr1=tr1,tr2=tr2)
                    binintmap = Crbin[0]*1000
                    binintmap[np.isnan(binintmap)]=-1000
                    binintmap = binintmap.astype("int32")
                    saveTIF(hsfolder +"/" + hsfile,Crbin,savefigfolder+"/binary/"+obsfile[:-4]+"binary_50.tif")
        
            except Exception as e:
                print("###################################################################")
                print("/!\/!\/file cannot being computed due to the following ERROR /!\/!\/")
                print("species concerned:",obsfile[:-4])
                print(e)
                print("###################################################################")
        else:
            print("file already computed")       

print("spatial methods imported")