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
import os
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
import community as co
from scipy.spatial.distance import pdist

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph
import igraph
import leidenalg
from scipy.sparse import coo_matrix

#######################################################################################################################################################################

def reassign_non_continuous_regions(arrayp):

    """Reassign non-continuous regions in a labeled array to new labels. Example: Ecoregions raster, has sometimes same ecoregions but separated by 
    a non-valid space (sea), this function label again the subregions but renaming them if two same iDs are not continuous
    
    Input: 
    
    array: (2-dimensional numpy array) containing iD (integer values), labelling different regions of the map. Example: regions of "1" next to regions of "2" 

    Output:

    array: (2-dimensional numpy array) containing iD (integer values), labelling different regions of the map. Example: regions of "1" next to regions of "2", but
    each region are necessary continuous. Example: two non contiguous regions of "1" are impossible, one of the region will be labelled as "3" or any other
    value nontaken by other region.

    
    """

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


#######################################################################################################################################################################

# get a 2D array from a shpfile and reproject it using a ref file
def project_shpfile(reftifpath,shpfile):

    """
    Rasterize a shapefile using a reference TIFF file.
    
    Inputs:

    reftifpath: (str), path to a refference .tif file to copy the projection, extent and resolution.
    shpfile: (path to a .shp) shape file to fill the map

    Output:

    raster: (2-dimensional numpy array)  representing the shape file with appropriate projection extent and resolution
    
    """

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

#######################################################################################################################################################################

def saveTIF(reffile, array, outputfile):

    """ Save a numpy array to a TIFF file using rasterio.
    
    Inputs:

    reffile: (str), path to a refference .tif file to copy the projection, extent and resolution.
    array: (2-dimensional numpy array), array to fill the .tif to be created
    outputfile: (str), name and path of the .tif output.

    Output:

    Save a numpy array to a TIFF file using rasterio.
    
    """


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

#######################################################################################################################################################################

def gkern(sig=1.,hole=False):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`

    Input:

    sig: (float, default = 1.) standard deviation of the Gaussian Kernel to be created
    hole: (bool, default=False) if True: a hole is drill at the center of the gaussian kernel, used in cases where we do not want
    an occurence density to influenced the occurence location itself.
    

    """
    l=int(8*sig + 1)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    if hole==True:
        kernel[kernel==np.max(kernel)]=0
    return kernel / np.sum(kernel)

#######################################################################################################################################################################

def circkernel(A):
    """
    Creates a circular kernel of specified area A, used to compute the coverage metric, itself used to compute weights in the subregions
    in the spatial envelope generation method called PseudoRange.

    Input:

    A:(float) Area of the disk to be created

    Output:

    K: (2-dimensional numpy array), a disk of ones surronded by zeros.
    """
    R=(A/np.pi)**(1/2)
    K=np.zeros((2*int(R)+1,2*int(R)+1))
    Ks=K.shape[0]
    mid=Ks//2
    for x in range(Ks):
        for y in range(Ks):
            if ((x-mid)**2 + (y-mid)**2)**(1/2) <= R:
                K[x,y]=1
    return K

#######################################################################################################################################################################

def applykernel_pr(Map,Kernel): 
        
        """ Apply a kernel to a map, considering only the pixels with value 1 in the map.
        
        Inputs:
        
        Map: (2-dimensional numpy array), map where the kernel would be applied on location with value 1
        Kernel: (2-dimensional numpy array), 2 dimensional probability distribution (usually gaussian)

        Output:
        
        """

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


#######################################################################################################################################################################

def ifelse(condition,resultif,resultelse):
    """
    standard ifelse

    Inputs:

    condition:(bool) boolean to be tested

    Output:

    resultif: result to return if the condition is True
    resultelse: result to return if the condition is False
    """
    if condition == True:
        return resultif
    else:
        return resultelse


#######################################################################################################################################################################

def gaussian_kernel(sig):

    """ Generate a Gaussian kernel with a given standard deviation (sig).
    
    Input:

    sig: (float) standard deviation (in pixels) of the gaussian kernel to be generated
    
    Output:

    K: (2-dimensional numpy array) 2D Gaussian kernel of standard deviation sig
    
    """

    size = int(8*sig + 1)
    K=np.zeros((size,size))
    mid=K.shape[0]//2
    for x in range(0,size):
        for y in range(0,size):
            K[x,y]=np.exp(-(((x-mid)**2)/(2*sig**2)+((y-mid)**2)/(2*sig**2)))
    K[mid,mid]=0
    return K/np.nansum(K)

# Estar algorithm permits to produce a ExpoScore Map based on observations and IUCN
IUCNalone=False

#######################################################################################################################################################################
 
def applykernel(Map,Kernel,NanMap=None,WeightMap=None): 
    """ Apply a kernel to a map, considering only the pixels with value 1 in the map, this function is used in PseudoRange method to compute
    subregions weighting.
    
    Inputs:
    
    Map:(2-dimensional numpy array), usually an Obs map, with most locations with value 0, and some with value 1 where the species has been observed
    Kernel:(2-dimensional numpy array), a kernel (2d-proability density distribution) usually created using function gkern(), to produce a 2D Gaussian kernel.
    NanMap:(2-dimensional numpy array), a map indicating where are irrelevent terrain type (out of study extent or sea for terrestrial animals)
    WeightMap:(2-dimensional numpy array), a map giving the weight associated to the applied kernel at each location. 

    Output:

    newMap:(2-dimensional numpy array), a new distribution obtained by applying the kernel to each location with value 1 in the Map input.
    
    """
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

#######################################################################################################################################################################

def lonlat2Obs(path2csv,path2ref_tif,loncolname,latcolname,spnamecolname,resolution,separator,speciesname=None):
    """
    Convert longitude and latitude from a CSV file to a raster array based on a reference TIFF file.
    
    Inputs:

    path2csv: (str) path to .csv file containing all observations of the species, the file should contain the name of the species in one column, 
    a column for longitude and a column for latitude.
    path2ref_tif: (str) path to a refference file .tif raster containing all required metadata to project the occurences on a 2-dimensional array, 
    the occurences should be extracted and projected in the same format as the reference file.
    loncolname: (str) name of the longitude column in the .csv file
    latcolname: (str) name of the latitude column in the .csv file
    spnamecolname: (str) name of the species name or code column in the .csv file
    resolution (str): number of meters represented in a pixel of the new projection (ex: 1000) if 1 pixel represents 1km x 1km square.
    separator (str): separator used by the .csv file (ex: "," , ";" , " " or tabulation)
    speciesname (str): if a specific name should be extracted (ex: "Acanthis_flammea")

    Output:

    raster_array: (2-dimensional numpy array) corresponding to the mapped occurences reported in the .csv file.
    
    """
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

#######################################################################################################################################################################

def estarformat(hspath, occurencecsvfile, loncolname, latcolname, spnamecolname, resolution, outputlocation,sep=","):
    """
    Fonction pour générer des rasters par espèce et les organiser dans un répertoire Estarbox.

    Inputs:
        hspath (str): Chemin vers le dossier contenant les fichiers .tif.
        occurencecsvfile (str): Chemin vers le fichier CSV des occurrences.
        loncolname (str): Nom de la colonne des longitudes dans le CSV.
        latcolname (str): Nom de la colonne des latitudes dans le CSV.
        spnamecolname (str): Nom de la colonne contenant les noms des espèces dans le CSV.
        resolution (float): Résolution pour lonlat2Obs.
        outputlocation (str): Chemin de destination pour le répertoire "Estarbox".

    Outputs:

    Generate a Estar folder with Obs HS and fill information in these folders based on the CSV files given

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

#######################################################################################################################################################################

def generate_density_map(Obs, resolution=0.5, max_obs = 10000, verbose=False):

    """ Generate a density map based on the nearest neighbor distance of observation points in a 2D binary array, it is a Kernel Density Estimation
    with a spatially varyinf kernel size based on local occurence density.
    
    Inputs:

    Obs: (2-dimensional numpy array) containing 0 everywhere, except where the species has been observed (value >=1)
    resolution: (float, default=0.5) parameter controlling the size of the detected community 
    max_obs: (float, default=10000) resample the observations if too much obervations are present in the Obs array, 
    permits to fasten the algorithm if it takes too long, lower max_obs if the algorithm takes too long, but increase 
    it for better results.
    verbose: (bool, default=False) permits to print different steps to check algorithm advancement.

    Outputs:

    Map: (2-diimensional numpy array) of observations density (D) but rescaled using the transformation D/(1+D).
    
    """
    ### Get coordinates of observation points (pixels)
    coords = np.column_stack(np.where(Obs > 0))
    Nobs = len(coords)

    if verbose:
        print('Generating map of nearest neighbor distance')
        
    ### Generate map of distance to closest neighbor
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean',algorithm='auto')  # or 'cosine', 'manhattan', etc.
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    nn_map = np.ones_like(Obs)*np.nan
    nn_map[(coords[:,0],coords[:,1])] = distances[:,1]

    if verbose:
        print('Computing spatial communities')
        
    ### Compute spatial communities
    coords_df = pd.DataFrame(coords,columns=['row','col'])
    
    if len(coords)>max_obs:
        coords_10k = np.unique(coords //10,axis=0)
        coords_df[['row_10','col_10']]=coords //10
    else:
        coords_10k = coords
        coords_df[['row_10','col_10']]=coords
    
    coords10_df = pd.DataFrame(coords_10k,columns=['row_10','col_10'])        

    # Compute median distance
    radius = np.nanpercentile(pdist(coords_10k, metric='euclidean'),50)
    
    # Create a sparse graph of neighbors within a maximum radius equel to median distance
    A = radius_neighbors_graph(coords_10k, radius=radius, mode='distance', include_self=False)
    A_coo = A.tocoo()
    del A

    # Create spatial graph
    N = len(coords_10k)
    edges = list(zip(A_coo.row, A_coo.col))
    weights = A_coo.data.tolist()
    G = igraph.Graph(n=N,edges=edges, directed=False)
    G.es['weight'] = weights

    del edges, weights, A_coo

    # Compute communities
    partition = leidenalg.find_partition(
        G,
        leidenalg.CPMVertexPartition,
        weights='weight',                # Optional if you have edge weights
        resolution_parameter=resolution         # Tune this!
    )
 
    coords10_df['label']=partition.membership
    coords_df = pd.merge(coords_df,coords10_df)
    del coords10_df, coords, coords_10k

    if verbose:
        print('Mapping spatial communities')
        
    community_array = np.zeros_like(Obs, dtype=int)
    community_array[coords_df['row'].tolist(),coords_df['col'].tolist()] = coords_df['label'].values+1

    del coords_df, partition, G

    Map=np.zeros_like(community_array).astype("float64")
    for iD in np.unique(community_array)[1:]:
        print("community iD=",iD)
        array=(community_array==iD)
        nb_points_commu = np.nansum(array)
        print("nb points in community iD ",iD," =",nb_points_commu)
        if nb_points_commu !=1: # si il n'y a qu'un seul point on ne peut pas calculer de distance!
            mind = nn_map[np.where(array)].mean()
            print("mean minimal distance in community iD",iD,"=",mind)
            K=gkern(sig=int(2*mind))
            K=K/np.max(K)
            E=applykernel(array,K)
            Map+=E
        else:
            print("Not enough points in community ",iD," to compute a specified distance, point considered as outlier")

    Map = Map / (1+Map)
    return Map


#######################################################################################################################################################################

# new version of allign that is able to deal with n multiple folder and align files in it
def allign2(listfolders,listnamesformat):
    """ Align files in multiple folders based on a common naming format. 

    Inputs:
    
    listfolders: (list of str) list of folders containing the files to allign.
    listnamesformat: (list of str) list of names formats that permit to allign names in different folder. The names should be in a form of a variable part
    and a constant part permitting to allign files. As an example: if a first folder contains files names ["Acanthis_flammea_Obs.tif","Elanus caeruleus_Obs.tif"]
    and a HS folder contains the names ["Elaneus_caeruleus_HSfile.tif","Acanthis flammea_HSfile.tif"], the common name is symbolized by "XxX", so the user should
    write for listnamesformat in that case ["XxX_Obs.tif","XxX_HSfile.tif"].
    
    Outputs:

    ordered_indexes: (list of list of indexes), giving in order the indexes corresponding in each folder. In our example in listnamesformat we should get: [[0,1][1,0]], 
    since the first file in the Obs folder corresponds to the second folder in the HS folder, this is due to the common part "Acanthis_flammea"=="Acanthis flammea",
    the function supports scientific names or other names separated by " " or "_". 
    
    """
    listnames = [os.listdir(folder) for folder in listfolders]
    # get all variable parts in order for all folders
    list_variable_parts=[]
    for k,format in enumerate(listnamesformat):
        #remove suffix and preffix parts to extract variable part (species names)
        size_preffix = format.index("XxX")
        suffix = format[size_preffix + 3:]
        size_suffix = len(suffix)
        # for each file in the folder concerned create a list of variable parts
        # we enforce "_" as a separator, so if " " appears it is changed to "_" to make it comparable across folders
        list_variable_parts.append([ (file[size_preffix : -size_suffix]).replace(" ","_") for file in listnames[k]])
        # we are going to produce a list of indexes to take files in the appropriate order when using functions along entire folders
    ordered_indexes=[[] for n in range(len(listfolders))]
    # for each variable part for the first folder, try to match a file corresponding to the variable part in other folders
    ##########################################################################################
    for idx_folder1, vpart in enumerate(list_variable_parts[0]):
        # for each vpart extracted to a filename in the first folder of lsitfolders
        #try to match another variable part in other folders
        listadds=[idx_folder1] #idx to add for folders
        try:
            # for all folders except the fisrt one in simulatneous way:
            for folderidx, folder in enumerate(listfolders[1:]): # for all other folders except the first one
                #print(folderidx)
                listadds.append(list_variable_parts[folderidx+1].index(vpart)) # find the idx of file that has the same variable part
        except Exception as error:
            print(error)
            print("No complete allignment found for variable part= ",vpart)
        
        if len(listadds) == len(listfolders): # si on a bien une correspondance de vpart dans chaque folder
            for idxfolder,add in enumerate(listadds): # on rajoute les valeurs des indices dans le ordered_indexes
                ordered_indexes[idxfolder].append(add)
                
    maxfiles = max( len(list_variable_parts[k]) for k in range(len(list_variable_parts)))
    nbfiles_with_no_allignment = maxfiles - len(ordered_indexes[0])


    if nbfiles_with_no_allignment>0:
        print(nbfiles_with_no_allignment, " files have no correspondance in all folders")
    else:
        print("All files alligned,  1 to 1 correspondance found for all files in Obs and HS folders")

    return ordered_indexes


#######################################################################################################################################################################



print("utilitary tools imported")


