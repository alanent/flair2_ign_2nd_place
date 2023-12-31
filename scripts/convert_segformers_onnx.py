import accelerate
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor,SegformerFeatureExtractor
import pandas as pd
import cv2
import numpy as np
import albumentations as aug
import random
import rasterio
from pathlib import Path
import splitfolders
import shutil
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import time
from torch.nn.functional import interpolate
import onnx
from onnxconverter_common import auto_mixed_precision, auto_convert_mixed_precision, float16

##############################################################################################################################
# CHECK CUDA DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

##############################################################################################################################
# load LABEL DICT AND FEATURE EXTRACTOR
def array_to_dict(array):
    dictionary = {}
    for i, item in enumerate(array):
        dictionary[i] = item
    return dictionary

classes = ['None','building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
id2label = array_to_dict(classes)
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
feature_extractor = SegformerFeatureExtractor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

##############################################################################################################################
# LOAD MODELS
pretrained_model_name_b5_sentinel_norm =  "./models/segformer_b5_rgb_norm_sentinel2-4e"
model_b5_sentinel_norm = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name_b5_sentinel_norm,
    id2label=id2label,
    label2id=label2id,
    reshape_last_stage=True,
    ignore_mismatched_sizes=True
)

pretrained_model_name_b5_igb_norm =  "./models/segformer_b5_igb_norm_aerial-8e+psdl"
model_b5_igb_norm = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name_b5_igb_norm,
    id2label=id2label,
    label2id=label2id,
    reshape_last_stage=True,
    ignore_mismatched_sizes=True
)

##############################################################################################################################
# EXPORT TO ONNX
model_b5_sentinel_norm  = model_b5_sentinel_norm.to(device)
dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
torch.onnx.export(model_b5_sentinel_norm, dummy_input, "model_b5_sentinel_norm.onnx", verbose=True)

model_b5_igb_norm  = model_b5_igb_norm.to(device)
dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
torch.onnx.export(model_b5_igb_norm, dummy_input, "model_b5_igb_norm.onnx", verbose=True)

##############################################################################################################################
# CONVERT TO ONNX FLOAT 16 // REDUCE THE SIZE OF THE MODEL BY 2
model = onnx.load("model_b5_sentinel_norm.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "model_b5_sentinel_norm_fp16.onnx")

model = onnx.load("model_b5_igb_norm.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "model_b5_igb_norm_fp16.onnx")
