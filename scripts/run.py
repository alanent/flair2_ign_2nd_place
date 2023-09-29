import os
import pandas as pd
import cv2
import numpy as np
import albumentations as aug
import random
import rasterio
from pathlib import Path
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import time
import onnx

##############################################################################################################################
# Spécifiez les fournisseurs d'exécution que vous souhaitez utiliser

providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

import onnxruntime as ort
ort.get_device()

ort_session_b5_sentinel_norm = ort.InferenceSession("./models/model_b5_sentinel_norm_fp16.onnx", providers=providers)
ort_session_b5_igb_norm = ort.InferenceSession("./models/model_b5_igb_norm_fp16.onnx", providers=providers)

##############################################################################################################################
# Get path and names of files

def lister_images(dossier):
    chemins_images = []
    noms_images = []

    for dossier_racine, sous_dossiers, fichiers in os.walk(dossier):
        for fichier in fichiers:
            # Vérifie si le fichier se termine par ".tif"
            if fichier.endswith(".tif"):
                chemin_image = os.path.join(dossier_racine, fichier)
                nom_image = fichier

                chemins_images.append(chemin_image)
                noms_images.append(nom_image)

    return chemins_images, noms_images


# Spécifiez le chemin du dossier contenant les images
dossier_images = "./data/"

# Obtenez la liste des chemins et noms d'images
chemins, noms = lister_images(dossier_images)

##############################################################################################################################
# Inference

debut = time.time()
ix=0
for chemin, nom in zip(chemins, noms):
    if ix % 400 == 0:
        fin = time.time()
        temps_execution = int(fin - debut)
        print("nombre d'executions : ", ix, f" // temps d'execution : {temps_execution} secondes")
    ix= ix+1


    # get raster file
    with rasterio.open(chemin) as src_img:
        image = src_img.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        igb = src_img.read([4,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        igb = igb.astype(np.float32)

    # aerial igb normalization
    MEAN = np.array([ 0.40987858, 0.45704361, 0.42254708])
    STD = np.array([ 0.15510736, 0.1782405 , 0.17575739])
    igb_norm = aug.Compose([aug.Normalize(mean=MEAN, std=STD)])
    image_igb_norm = igb_norm(image=igb)['image']


    # sentinel normalization
    sen = np.load('./data/'+ nom.replace('IMG','SEN').replace('tif','npy'))

    #get MEAN and STD for normalization
    MEAN = np.mean(sen.squeeze(), axis=(1, 2))
    STD= np.std(sen.squeeze(), axis=(1, 2))

    #Check for NaN values
    nan_present = any(math.isnan(x) for x in MEAN)
    if nan_present:
        MEAN = np.array([0.44050665, 0.45704361, 0.42254708])
        STD = np.array([0.20264351, 0.1782405 , 0.17575739])
    sentinel_norm = aug.Compose([aug.Normalize(mean=MEAN, std=STD)])
    image_sentinel_norm = sentinel_norm(image=image)['image']

    pixel_values_sentinel_norm = feature_extractor(image_sentinel_norm, return_tensors="np").pixel_values.astype(np.float16) #.to(device)
    pixel_values_igb_norm = feature_extractor(image_igb_norm, return_tensors="np").pixel_values.astype(np.float16) #.to(device)


    # image_np = pixel_values.cpu().numpy()
    output_b5_sentinel_norm = ort_session_b5_sentinel_norm.run(None,  {"input.1":pixel_values_sentinel_norm})[0]
    output_b5_igb_norm = ort_session_b5_igb_norm.run(None,  {"input.1":pixel_values_igb_norm})[0]


    ######################################################################################################
    # aggregate results predictions
    pred_segformer_b5_igb_norm = tf.image.resize(tf.transpose(output_b5_igb_norm, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512
    pred_segformer_b5_sentinel_norm = tf.image.resize(tf.transpose(output_b5_sentinel_norm, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512


    preds = np.mean(np.array([ pred_segformer_b5_igb_norm,pred_segformer_b5_sentinel_norm  ]), axis = 0)
    preds = [np.argmax(preds[index,:,:,:], axis = -1).transpose((0,1)) for index in range(preds.shape[0])]
    preds = np.squeeze(preds)
    preds = np.array(preds)-1
    preds = preds.astype('uint8')  # Pass prediction on CPU


    preds_sent = np.mean(np.array([pred_segformer_b5_sentinel_norm]), axis = 0)
    preds_sent = [np.argmax(preds_sent[index,:,:,:], axis = -1).transpose((0,1)) for index in range(preds_sent.shape[0])]
    preds_sent =np.squeeze(preds_sent)
    preds_sent = np.array(preds_sent)-1
    preds_sent = preds_sent.astype('uint8')  # Pass prediction on CPU

    preds_igb = np.mean(np.array([pred_segformer_b5_igb_norm]), axis = 0)
    preds_igb  = [np.argmax(preds_igb [index,:,:,:], axis = -1).transpose((0,1)) for index in range(preds_igb.shape[0])]
    preds_igb  =np.squeeze(preds_igb )
    preds_igb  = np.array(preds_igb )-1
    preds_igb  = preds_igb.astype('uint8')  # Pass prediction on CPU

    preds[preds == 5] = preds_igb[preds == 5]
    preds[preds == 8] = preds_sent[preds == 8]
    preds[preds == 9] = preds_igb[preds == 9]
    preds[preds == 10] = preds_igb[preds == 10]
    preds[preds == 11] = preds_igb[preds == 11]


    chemin_sortie = "./preds/" + nom.split('/')[-1].replace('IMG', 'PRED')
    Image.fromarray(preds).save(chemin_sortie, compression='tiff_lzw')

fin = time.time()
temps_execution = int((fin - debut)/60)

print(f" Temps total d'execution : {temps_execution} minutes")
