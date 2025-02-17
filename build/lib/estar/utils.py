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
from shapely.geometry import Point
from rasterio.features import rasterize

import networkx as nx
from scipy.spatial.distance import cdist
import community as co
from scipy.spatial.distance import pdist

############################################# OLD ##########################

def reassign_non_continuous_regions(arrayp):
    array=arrayp.copy()
    unique_labels = np.unique(array)
    current_max_label = np.nanmax(unique_labels)
    
    for label_value in unique_labels:
        if label_value is None:
            continue
        
        # Create a binary mask for the current label
        mask = array == label_value
        
        # Label connected components in the mask
        labeled_mask, num_features = label(mask)
        
        if num_features > 1:
            for i in range(2, num_features + 1):
                current_max_label += 1
                #print(current_max_label)
                array[labeled_mask == i] = current_max_label
                
    return array        


# get a 2D array from a shpfile and reproject it using a ref file
def project_shpfile(reftifpath,shpfile):
    # Chemin vers le fichier TIFF
    tiff_path = reftifpath
    with rasterio.open(tiff_path) as src:
        reference_crs = src.crs
        reference_transform = src.transform
        width = src.width
        height = src.height
        bounds = src.bounds
    ecoregions = gpd.read_file(shpfile)
    ecoregions = ecoregions.to_crs(reference_crs)
    # Créer une liste de (geometry, value) pour chaque écorégion
    shapes = [(geom, float(i + 1)) for i, geom in enumerate(ecoregions.geometry)]
    # Définir la taille du raster
    out_shape = (height, width)
    # Rasteriser les géométries du shapefile
    raster = rasterize(shapes, out_shape=out_shape, transform=reference_transform, fill=0.0, dtype='float32')
    return raster



def saveTIF(reffile, array, outputfile):
    # Load reference metadata
    with rasterio.open(reffile) as ref:
        meta = ref.meta.copy()
        transform = ref.transform
        crs = ref.crs

    # Ensure the array is in int32 for efficient memory storage
    data = array.astype("int32")

    # Update metadata
    meta.update({
        "driver": "GTiff",
        "dtype": "int32",  # Ensure the data type matches your array
        "nodata":-9999,
        "height": data.shape[0],
        "width": data.shape[1],
        "transform": transform,
        "crs": crs,
        'count': 1,  # Number of bands
        'compress': 'LZW'  # Compression for smaller file size
    })

    # Save the numpy array to a new TIFF file
    with rasterio.open(outputfile, 'w', **meta) as dst:
        dst.write(data, 1)  # Write to the first band


def gkern(sig=1.,hole=False):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    l=int(8*sig + 1)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    if hole==True:
        kernel[kernel==np.max(kernel)]=0
    return kernel / np.sum(kernel)

def circkernel(A):
    R=(A/np.pi)**(1/2)
    K=np.zeros((2*int(R)+1,2*int(R)+1))
    Ks=K.shape[0]
    mid=Ks//2
    for x in range(Ks):
        for y in range(Ks):
            if ((x-mid)**2 + (y-mid)**2)**(1/2) <= R:
                K[x,y]=1
    return K

# def applykernel(Map,Kernel): 
#     imax,jmax = Map.shape
#     ks=Kernel.shape[0]
#     lx,ly = np.where(Map==1)
#     newMap=np.zeros((imax,jmax))
#     for x,y in zip(lx,ly):
#         x1=max(x-ks//2,0) ; x2 = min(x+ks//2+1,imax) # ça c'est bon
#         y1=max(y-ks//2,0) ; y2 = min(y+ks//2+1,jmax) # ça c'est bon
#         s1=ifelse(x-ks//2<0,abs(x-ks//2),0) ; s2 = ifelse(x+ks//2 >imax-1, (ks)-(x+ks//2-imax+1),ks)
#         r1=ifelse(y-ks//2<0,abs(y-ks//2),0) ; r2 = ifelse(y+ks//2 >jmax-1,(ks)-(y+ks//2-jmax+1),ks)
#         newMap[x1:x2,y1:y2] += Kernel[s1:s2,r1:r2] # concerned kernel part
#     return newMap


def cosgkern(sig=1.,hole=False):
    K=np.zeros((8*int(sig)+1,8*int(sig)+1))
    ks = K.shape[0]
    mid = ks//2
    for x in range(ks):
        for y in range(ks):
            d= ((x-mid)**2 + (y-mid)**2)**(1/2)
            K[x,y] = d
    K = np.cos(K*np.pi/(4*sig))*np.exp(-K**2/(2*sig**2))
    return K


def applykernel_pr(Map,Kernel): 
        imax,jmax = Map.shape
        ks=Kernel.shape[0]
        lx,ly = np.where(Map==1)
        newMap=np.zeros((imax,jmax))
        for x,y in zip(lx,ly):
            x1=max(x-ks//2,0) ; x2 = min(x+ks//2+1,imax) # ça c'est bon
            y1=max(y-ks//2,0) ; y2 = min(y+ks//2+1,jmax) # ça c'est bon
            s1=ifelse(x-ks//2<0,abs(x-ks//2),0) ; s2 = ifelse(x+ks//2 >imax-1, (ks)-(x+ks//2-imax+1),ks)
            r1=ifelse(y-ks//2<0,abs(y-ks//2),0) ; r2 = ifelse(y+ks//2 >jmax-1,(ks)-(y+ks//2-jmax+1),ks)
            newMap[x1:x2,y1:y2] += Kernel[s1:s2,r1:r2] # concerned kernel part
        return newMap





def Beta2(x,mu,var):
    alpha=((mu*(1-mu))/var-1)*mu
    beta=((mu*(1-mu))/var-1)*(1-mu)
    dist = scipy.stats.beta(alpha, beta)
    return dist.pdf(x)

# generate an exponential kernel based on Mean Dispersal Distance, return a 2D exponential kernel generated from an exponential
# distance distribution with mean =MDD, hole paramter remove the centered pixel and its neighbors, sizecoeff is the parameter  
# that control the size of the generated kernel, higher values for sizecoeff means generation of a larger part of the kernel

def ifelse(condition,resultif,resultelse):
    if condition == True:
        return resultif
    else:
        return resultelse

def gaussian_kernel(sig):
    size = int(8*sig + 1)
    K=np.zeros((size,size))
    mid=K.shape[0]//2
    for x in range(0,size):
        for y in range(0,size):
            K[x,y]=np.exp(-(((x-mid)**2)/(2*sig**2)+((y-mid)**2)/(2*sig**2)))
    K[mid,mid]=0
    return K/np.nansum(K)

def generate_exponential_kernel(MDD,hole=True,sizecoeff=10,size=None):
    if size is None:
        size = 2*(int(MDD*sizecoeff//2)) +1 # value 20 is high but we need to compute a large area to not having jump on the map due to lack of border calculation
    mid=size // 2
    X = np.arange(size)
    Y = np.arange(size)
    kernel=np.zeros((size,size))
    for x in X:
        for y in Y:
            distance_squared = (x-mid)**2 + (y-mid)**2
            kernel[x,y] = np.exp(-np.sqrt(distance_squared) * (1 / MDD))
    if hole==True:
        kernel[mid-1:mid+2,mid-1:mid+2]=0
    return kernel

# Estar algorithm permits to produce a ExpoScore Map based on observations and IUCN
IUCNalone=False


# Function to find the bin index for a given number
def get_bin_index(number, bin_edges):
    """
    Given a number and bin edges, returns the index of the bin the number falls into.
    If the number is outside the range, returns None.
    """
    if number < bin_edges[0] or number > bin_edges[-1]:
        return None
    # np.digitize returns bins in 1-based index, so we subtract 1
    bin_index = np.digitize(number, bin_edges) - 1
    # Handle the edge case where the number is exactly equal to the maximum edge
    if bin_index == len(bin_edges) - 1:
        bin_index -= 1
    return bin_index

# Function to get the relative frequency of the bin a number falls into
def get_relative_frequency(number, bin_edges, relative_frequencies):
    bin_index = get_bin_index(number, bin_edges)
    if bin_index is None:
        return 0  # Number is outside the range
    return relative_frequencies[bin_index]


def genhist(L,num_bins=10):
    # Compute histogram
    counts, bin_edges = np.histogram(L, bins=num_bins, range=(min(L), max(L)), density=False)
    # Calculate relative frequencies
    total_count = len(L)
    relative_frequencies = counts / total_count
    return bin_edges,relative_frequencies
 
def applykernel(Map,Kernel,NanMap=None,WeightMap=None): 
    imax,jmax = np.shape(Map)
    ks=Kernel.shape[0]
    lx,ly = np.where(Map==1)
    newMap=np.zeros((imax,jmax))
    if NanMap is not None:  
        continuous_regions=1-NanMap
        iD_continuous_regions = label(continuous_regions, structure=np.ones((3, 3)))[0]
    else:
        iD_continuous_regions = np.ones_like(Map)
    
    inhabited_regions = np.unique(iD_continuous_regions[np.where(Map==1)])
    print("list of inhabited continuous regions iD",inhabited_regions)
    print("applying kernels...")
    nb_points_done =0 
    nb_points = len(lx)
    t0=Time.time()
    for region in inhabited_regions:
        if Time.time()-t0 > 5:
            print(nb_points_done ,"/", nb_points, " points done")
            t0 = Time.time()
        Mask_accessible_region = iD_continuous_regions==region
        xs,ys = np.where(Map*Mask_accessible_region)
        for x,y in zip(xs,ys):
            if WeightMap is not None:
                weight = WeightMap[x,y]
            else:
                weight = 1
            x1=max(x-ks//2,0) ; x2 = min(x+ks//2+1,imax) # ça c'est bon
            y1=max(y-ks//2,0) ; y2 = min(y+ks//2+1,jmax) # ça c'est bon
            s1=ifelse(x-ks//2<0,abs(x-ks//2),0) ; s2 = ifelse(x+ks//2 >imax-1, (ks)-(x+ks//2-imax+1),ks)
            r1=ifelse(y-ks//2<0,abs(y-ks//2),0) ; r2 = ifelse(y+ks//2 >jmax-1,(ks)-(y+ks//2-jmax+1),ks)
            newMap[x1:x2,y1:y2] += Kernel[s1:s2,r1:r2]*Mask_accessible_region[x1:x2,y1:y2]*weight # concerned kernel part
            nb_points_done +=1
    return newMap


# should be in the format ; or , as separator and "." marker for decimals

def lonlat2Obs(path2csv,path2ref_tif,loncolname,latcolname,spnamecolname,resolution,separator,speciesname=None):
    # Étape 1: Lire le fichier CSV avec pandas
    csv_file = path2csv  # remplace par le chemin de ton fichier CSV
    print("trying import")
    data = pd.read_csv(csv_file,sep=separator)
    print("imported")
    if speciesname is not None:
        data=data[data[spnamecolname]==speciesname]
    # On suppose que tes colonnes de longitude et latitude s'appellent 'longitude' et 'latitude'
    longitudes = data[loncolname].values
    latitudes = data[latcolname].values
    print("extract latitudes and longitudes")
    # Étape 2: Lire les métadonnées du fichier TIFF avec rasterio
    tiff_file =  path2ref_tif # remplace par le chemin de ton fichier TIF
    df = pd.DataFrame()
    df[loncolname]=longitudes
    df[latcolname]=latitudes
    # Step 2: Create geometry column from longitude and latitude
    df['geometry'] = [Point(xy) for xy in zip(df[loncolname], df[latcolname])]
    # Step 3: Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    print("conversion to spatial points")
    # Step 4: Set the coordinate reference system (CRS) to WGS84 (EPSG:4326) for lat/lon
    gdf.set_crs(epsg=4326, inplace=True)
    # Print the resulting GeoDataFrame
    with rasterio.open(tiff_file) as dataset:
        # Get the CRS (Coordinate Reference System)
        crs = dataset.crs
        bounds = list(dataset.bounds)
    gdf_reproj = gdf.to_crs(epsg=3035)
    # Étendue du raster (xmin, ymin, xmax, ymax) - exemple pour une zone en EPSG:32633 (UTM Zone 33N)
    extent = bounds # xmin, ymin, xmax, ymax
    # Résolution (taille des pixels en mètres, par exemple 100 mètres)
    # Dimensions du raster (nombre de lignes et de colonnes)
    width = int((extent[2] - extent[0]) / resolution)  # Nombre de colonnes
    height = int((extent[3] - extent[1]) / resolution)  # Nombre de lignes
    # Créer un tableau numpy rempli de zéros
    print("initialize a new raster")
    raster_array = np.zeros((height, width), dtype=np.uint8)
    # Transformer (projection linéaire entre coord géo et grille)
    transform = from_bounds(*extent, width, height)
    # Exemple de points géométriques en coordonnées projetées (EPSG:32633)
    points = gdf_reproj['geometry']
    # Parcourir les points et les placer dans le tableau
    print("total points = ",len(points))
    t0=Time.time()
    n=0
    for point in points:
        n+=1
        if Time.time()-t0>5:
            print(str(n)+"/"+str(len(points))+" done")
            t0=Time.time()
        # Convertir les coordonnées du point en indices de la matrice
        row, col = ~transform * (point.x, point.y)  # Inverser la transformation pour obtenir (row, col)
        row, col = int(col), int(row)  # Convertir en entiers (indices)

        # Vérifier si le point est dans l'étendue
        if 0 <= row < height and 0 <= col < width:
            raster_array[row, col] = 1  # Mettre un "1" à cet emplacement

    # Afficher la matrice résultante
    return raster_array



############################## Get all input format

import os
import pandas as pd

def estarformat(hspath, occurencecsvfile, loncolname, latcolname, spnamecolname, resolution, outputlocation,sep=","):
    """
    Fonction pour générer des rasters par espèce et les organiser dans un répertoire Estarbox.

    Args:
        hspath (str): Chemin vers le dossier contenant les fichiers .tif.
        occurencecsvfile (str): Chemin vers le fichier CSV des occurrences.
        loncolname (str): Nom de la colonne des longitudes dans le CSV.
        latcolname (str): Nom de la colonne des latitudes dans le CSV.
        spnamecolname (str): Nom de la colonne contenant les noms des espèces dans le CSV.
        resolution (float): Résolution pour lonlat2Obs.
        outputlocation (str): Chemin de destination pour le répertoire "Estarbox".
    """
    # Créer le dossier Estarbox et ses sous-dossiers HS et Obs
    estarbox_dir = os.path.join(outputlocation, "Estarbox")
    hs_dir = os.path.join(estarbox_dir, "HS")
    obs_dir = os.path.join(estarbox_dir, "Obs")
    os.makedirs(hs_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)

    # Charger le fichier CSV des occurrences
    occurences = pd.read_csv(occurencecsvfile,sep=sep)

    # Extraire les noms d'espèces uniques
    species_names = occurences[spnamecolname].unique()

    # Pour chaque espèce unique, générer le raster
    for species in species_names:
        # Chemin vers le fichier .tif correspondant à l'espèce
        ref_tif = os.path.join(hspath, os.listdir(hspath)[0])

        # Vérifier si le fichier .tif existe
        if not os.path.exists(ref_tif):
            print(f"Attention : fichier .tif manquant pour {species}, chemin attendu : {ref_tif}")
            continue

        # Appeler lonlat2Obs pour générer l'array correspondant
        obs_array = lonlat2Obs(
            path2csv=occurencecsvfile,
            path2ref_tif=ref_tif,
            loncolname=loncolname,
            latcolname=latcolname,
            spnamecolname=spnamecolname,
            resolution=resolution,
            speciesname=species,
            separator=sep)

        # Chemin de sauvegarde pour le raster de l'espèce
        output_tif_path = os.path.join(obs_dir, f"{species}.tif")

        # Sauvegarder le raster en utilisant saveTIF
        saveTIF(reffile=ref_tif, array=obs_array, outputfile=output_tif_path)

        print(f"Fichier enregistré pour {species} : {output_tif_path}")

    print(f"Traitement terminé. Les fichiers sont disponibles dans le dossier : {estarbox_dir}")


############################################## DENSITY MAP LIOUVAIN NETWORK #########################################



# Step 1: Extract occurrence coordinates
def get_occurrence_coordinates(array):
    return np.column_stack(np.where(array == 1))

# Step 2: Build a distance-based network
def build_network(coords, distance_threshold):
    distances = cdist(coords, coords, metric='euclidean')
    G = nx.Graph()
    for i, coord in enumerate(coords):
        G.add_node(i, pos=tuple(coord))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if distances[i, j] <= distance_threshold:
                G.add_edge(i, j, weight=distances[i, j])
    return G

# Step 3: Apply Louvain algorithm
def detect_communities(G):
    partition = co.community_louvain.best_partition(G)
    return partition

# Step 4: Assign community IDs back to the 2D array
def assign_communities_to_array(array, coords, communities):
    community_array = np.zeros_like(array, dtype=int)
    for node_id, community_id in communities.items():
        coord = coords[node_id]
        community_array[tuple(coord)] = community_id + 1  # Start community IDs from 1
    return community_array

# Step 5: Plot the network
def plot_network(G, communities, coords,background=None):
    pos = nx.get_node_attributes(G, 'pos')
    for e in pos:
        x,y=pos[e]
        pos[e]=(y,x)
    node_colors = [communities[node] for node in G.nodes]
    plt.figure(figsize=(8, 8))
    plt.title("Distance-Based Network with Communities")
    if background is not None:
        plt.imshow(background,cmap="Greys")
        plt.colorbar(shrink=0.7)
    nx.draw(G, pos, node_color=node_colors, with_labels=False, cmap=plt.cm.Set3, node_size=10, edge_color="grey")
    plt.show(block=False)

# Main workflow
def main(array, distance_threshold,background=None,plot=True):
    Nobs=np.nansum(array>=1)
    coords = get_occurrence_coordinates(array)
    G = build_network(coords, distance_threshold)
    communities = detect_communities(G)
    community_array = assign_communities_to_array(array, coords, communities)
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_component = max(components, key=len)
    size_largest_component = len(largest_component)
    print("number of components =",num_components)
    percentage= size_largest_component/Nobs*100
    print("largest component =",size_largest_component," points (",round(percentage,2),"%)")
    #if plot==True:
    #    plot_network(G, communities, coords,background=background)
    return community_array,percentage,num_components


def find_50percentile(array):
    # Step 1: Extract coordinates of points where the value is 1
    points = np.argwhere(array == 1)
    # Step 2: Compute all pairwise Euclidean distances
    pairwise_distances = pdist(points, metric='euclidean')
    # Step 3: Compute the 95th percentile of the distances
    percentile_50 = np.percentile(pairwise_distances, 50)
    print("median distance = ",percentile_50)
    return percentile_50



def find_distance(Obs,background=None,plot=False,bynx=10,comp1percent=50):
    xrow,ycol = np.shape(Obs)
    distrange=np.arange(1,xrow//2,bynx)
    distance_threshold = find_50percentile(Obs)
    community_array,percentage,num_components = main(Obs, distance_threshold,background=background,plot=plot)
    return community_array,percentage,num_components
    

def mean_minimum_distance(array):
    # Step 1: Extract coordinates of occurrences
    coords = np.column_stack(np.where(array == 1))
    
    # If there are fewer than 2 points, return 0 (no distances to compute)
    if len(coords) < 2:
        return 0.0
    
    # Step 2: Compute pairwise distances
    distances = cdist(coords, coords, metric='euclidean')
    
    # Step 3: For each point, find the minimum distance to another point (exclude self-distances)
    np.fill_diagonal(distances, np.inf)  # Exclude self-distances by setting diagonal to infinity
    min_distances = distances.min(axis=1)
    
    # Step 4: Compute the mean of the minimum distances
    mean_min_distance = np.percentile(min_distances,50)
    return mean_min_distance


def Pxy(Obs,background=None,plot=False,bynx=10,comp1percent=50,sigdefault=40):
    commu,perc,comp=find_distance(Obs,background=background,plot=plot,bynx=bynx,comp1percent=comp1percent)
    Map=np.zeros_like(commu).astype("float64")
    for iD in np.unique(commu)[1:]:
        print("community iD=",iD)
        array=commu==iD
        nb_points_commu = np.nansum(array)
        print("nb points in community iD ",iD," =",nb_points_commu)
        if nb_points_commu !=1: # si il n'y a qu'un seul point on ne peut pas calculer de distance!
            mind = mean_minimum_distance(array)
            print("mean minimal distance in community iD",iD,"=",mind)
            K=gkern(sig=int(2*mind))
            K=K/np.max(K)
            E=applykernel(array,K)
            Map+=E
        else:
            print("Not enough points in community ",iD," to compute a specified distance, point considered as outlier")
       
    Map = Map/(1+Map) # credibility transform
    if plot==True:
        plt.figure(figsize=(10,10))
        if background is not None:
            plt.imshow(background,cmap="Greys")
        alpha=Map/np.nanmax(Map)
        alpha[np.isnan(alpha)]=0
        plt.imshow(Map,cmap="inferno",alpha=alpha)
        plt.colorbar(shrink=0.7)
    return Map



def allign(obsfolder,hsfolder,listnamesformat):
    obs_example , hs_example = listnamesformat
    listobsnames = os.listdir(obsfolder)
    listhsnames = os.listdir(hsfolder)
    # get preffix sizes
    obs_size_prefix = obs_example.index("XxX") ; hs_size_prefix = hs_example.index("XxX")
    # get suffix
    obs_suffix = obs_example[obs_size_prefix + 3:]  ; hs_suffix = hs_example[hs_size_prefix +3:]
    # get suffix sizes
    obs_size_suffix = len(obs_suffix) ; hs_size_suffix = len(hs_suffix)
    # for both folders, get the list in order of variables parts
    list_variable_parts=[[],[]]
    # get all variable parts for obs
    list_variable_parts[0] = [ obsname[obs_size_prefix : -obs_size_suffix] for obsname in listobsnames]
    list_variable_parts[1] = [ hsname[hs_size_prefix : -hs_size_suffix] for hsname in listhsnames]
    
    obs_ordered_index=[]
    hs_ordered_index=[]

    # test if the variable part format corresponds "_" or " " between Genera and species names
    #########################################################################################
    if "_" in list_variable_parts[0][0] and " " in list_variable_parts[1][0]:
        for k,e in enumerate(list_variable_parts[1]):
            list_variable_parts[1][k]=e.replace(" ","_")

    if " " in list_variable_parts[0][0] and "_" in list_variable_parts[1][0]:
        for k,e in enumerate(list_variable_parts[1]):
            list_variable_parts[1][k]=e.replace("_"," ")
    ##########################################################################################

        
    for n,variable_part in enumerate(list_variable_parts[0]):
        # try if a corresponding index exists
        try:
            hs_ordered_index.append(list_variable_parts[1].index(variable_part))
            obs_ordered_index.append(n)
        except Exception:
            print("No file corresponds in HS folder for ",variable_part)

    nbfiles_with_no_allignment = max(len(list_variable_parts[0]),len(list_variable_parts[1])) - len(hs_ordered_index)
    if nbfiles_with_no_allignment>0:
        print(nbfiles_with_no_allignment, " files have no correspondance (obs or hs missing)")
    else:
        print("All files alligned,  1 to 1 correspondance found for all files in Obs and HS folders")
        
    return obs_ordered_index,hs_ordered_index




print("utilitary tools imported")


