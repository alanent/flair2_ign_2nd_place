import os
import numpy as np
from pathlib import Path
import json
from random import shuffle
import cv2
import random
import time
import torch
import datetime
import torchvision.transforms as T
import rasterio
import os
import re
import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import hex2color
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
import rasterio
import rasterio.plot as plot
import torch
import torchvision.transforms as T
import datetime
import time
import albumentations as aug
import pandas as pd
import yaml

def get_data_paths(path, filter):
    for path in Path(path).rglob(filter):
         yield path.resolve().as_posix()


def load_data (config: dict, val_percent=0.8):
    """ Returns dicts (train/val/test) with 6 keys:
    - PATH_IMG : aerial image (path, str)
    - PATH_SP_DATA : satellite image (path, str)
    - PATH_SP_DATES : satellite product names (path, str)
    - PATH_SP_MASKS : satellite clouds / snow masks (path, str)
    - SP_COORDS : centroid coordinate of patch in superpatch (list, e.g., [56,85])
    - PATH_LABELS : labels (path, str)
    - MTD_AERIAL: aerial images encoded metadata
    """
    def get_data_paths(config: dict, path_domains: str, paths_data: dict, matching_dict: dict, test_set=False) -> dict:
        #### return data paths
        def list_items(path, filter):
            for path in Path(path).rglob(filter):
                yield path.resolve().as_posix()
        status = ['train' if test_set == False else 'test'][0]
        ## data paths dict
        data = {'PATH_IMG':[], 'PATH_SP_DATA':[], 'SP_COORDS':[], 'PATH_SP_DATES':[],  'PATH_SP_MASKS':[], 'PATH_LABELS':[], 'MTD_AERIAL':[]}
        for domain in path_domains:
            for area in os.listdir(Path(paths_data['path_aerial_'+status], domain)):
                aerial = sorted(list(list_items(Path(paths_data['path_aerial_'+status])/domain/Path(area), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
                sen2sp = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*data.npy')))
                sprods = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*products.txt')))
                smasks = sorted(list(list_items(Path(paths_data['path_sen_'+status])/domain/Path(area), '*masks.npy')))
                coords = []
                for k in aerial:
                    coords.append(matching_dict[k.split('/')[-1]])
                data['PATH_IMG'] += aerial
                data['PATH_SP_DATA'] += sen2sp*len(aerial)
                data['PATH_SP_DATES'] += sprods*len(aerial)
                data['PATH_SP_MASKS'] += smasks*len(aerial)
                data['SP_COORDS'] += coords
                if test_set == False:
                    data['PATH_LABELS'] += sorted(list(list_items(Path(paths_data['path_labels_'+status])/domain/Path(area), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        # if config['aerial_metadata'] == True:
        #     data = adding_encoded_metadata(config['data']['path_metadata_aerial'], data)

        return data

    paths_data = config['data']
    with open(paths_data['path_sp_centroids'], 'r') as file:
        matching_dict = json.load(file)
    path_trainval = Path(paths_data['path_aerial_train'])
    train_domains = os.listdir(path_trainval)
    shuffle(train_domains)

    dict_train = get_data_paths(config, train_domains, paths_data, matching_dict, test_set=False)

    return dict_train


def filter_dates(img, mask, clouds:bool=2, area_threshold:float=0.2, proba_threshold:int=20):
    """ Mask : array T*2*H*W
        Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
        Area_threshold : threshold on the surface covered by the clouds / snow
        Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)

        Return array of indexes to keep
    """
    dates_to_keep = []

    for t in range(mask.shape[0]):
        # Filter the images with only values above 1
        c = np.count_nonzero(img[t, :, :]>1)
        if c != img[t, :, :].shape[1]*img[t, :, :].shape[2]:
            # filter the clouds / snow
            if clouds != 2:
                cover = np.count_nonzero(mask[t, clouds, :,:]>=proba_threshold)
            else:
                cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
            cover /= mask.shape[2]*mask.shape[3]
            if cover < area_threshold:
                dates_to_keep.append(t)
    return dates_to_keep

def read_dates(txt_file: str) -> np.array:
    with open(txt_file, 'r') as f:
        products= f.read().splitlines()
    dates_arr = []
    for file in products:
        dates_arr.append(datetime.datetime(2021, int(file[15:19][:2]), int(file[15:19][2:])))
    return np.array(dates_arr)


####################################################################################################################################"
#GET DATA LOCATION

config_path = "./data/flair-2-config.yml" # Change to yours
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# Creation of the train, val and test dictionnaries with the data file paths
d_train = load_data(config)


images = d_train["PATH_IMG"]
labels = d_train["PATH_LABELS"]
sentinel_images = d_train["PATH_SP_DATA"]
sentinel_masks = d_train["PATH_SP_MASKS"] # Cloud masks
sentinel_products = d_train["PATH_SP_DATES"] # Needed to get the dates of the sentinel images
centroids = d_train["SP_COORDS"] # Position of the aerial image in the sentinel super area



indices= range (0, len(images))

# data = { 'MEAN': [], 'STD': [],  }
df = pd.DataFrame()

img_to_plot = []
for u, idx in enumerate(indices):

    print(idx)
    sen = np.load(sentinel_images[idx])[:,[2,1,0],:,:]/2000
    mask = np.load(sentinel_masks[idx])
    dates_to_keep = filter_dates(sen, mask)
    sen = sen[dates_to_keep]
    # print(sen.shape)
    sen = np.mean(sen, axis=0)
    # print(sen.shape)
    # Transpose the array dimensions to (207, 207, 3)


    # Extraire le nom de fichier après le dernier '/'
    nom_fichier = images[idx].split('/')[-1]

    # Remplacer le préfixe "IMG" par "SEN"
    nouveau_nom = nom_fichier.replace("IMG", "SEN").replace(".tif", "")

    sen_spatch = sen[:, centroids[idx][0]-int(20):centroids[idx][0]+int(20),centroids[idx][1]-int(20):centroids[idx][1]+int(20)]
    transform = T.CenterCrop(10)
    sen_aerialpatch = transform(torch.as_tensor(np.expand_dims(sen_spatch, axis=0))).numpy()


    # Chemin complet de sauvegarde
    chemin_sauvegarde = "./data/sentinel/" + nouveau_nom
    # Sauvegarder le tableau NumPy
    np.save(chemin_sauvegarde, sen_aerialpatch)
