# IGN FLAIR2 Solution
Ce référentiel contient les résultats de notre participation au concour FLAIR2 organisé par l'IGN.
- Lien du concours : https://codalab.lisn.upsaclay.fr/competitions/13447#results
- Source des scripts : https://huggingface.co/blog/fine-tune-segformer
- Source de données : https://ignf.github.io/FLAIR/#FLAIR2


## Entraînement
Nous avons entrainé 4 segformers avec 4 pré-processing et données différentes. Mis à part ces traitements, nous n'avons pas effectué d'augmentations de données supplémentaires.
- Pour l'utilisation des données sentinels-2, nous avons filtrer les données contenant des nuages et de la neige. Nous enregistrons ensuite les données Sentinel2 correpondantes à chaque image aérienne `IMG_{Img_id}.npy` dans un fichier appelé `SEN_{Img_id}.npy`. Pour chaque image aérienne `IMG_{Img_id}.npy`, nous normalisons cette image par les moyennes et les écart-types de chaque chanaux (rgb) de l'image sentinel2 `SEN_{Img_id}.npy` précédement créée.
- Pour la normalisation imagenet, nous avons repris le preprocessing effectué par la classe `SegformerFeatureExtractor()` de HuggingFace.
- Pour la normalisation des images aeriennes nous utilisons, les moyennes et les écart-types de chaque canaux de l'ensemble du dataset des images aériennes.

| Modèle                           | GPU d'Entraînement  | Lot d'Entraînement | Nombre d'Époques | Mean IoU (test)  | Models Link      | Training Script  |
|----------------------------------|---------------------|--------------------|------------------|------------------|------------------|------------------|
| SegFormer-B5 RGB Norm Sentinel2  | NVIDIA Tesla A100   | 8                  | 4                | 61.3             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1GfI5OwrInzdMz--_VC0jY5AU13bxRJoy?usp=sharing)                 | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Cwkb2vIiXiXxhZaZ-98Cuusne_mp4KX?authuser=1#scrollTo=LDZvoduQLNjI)                |
| SegFormer-B5 RGB Norm Aerial     | NVIDIA Tesla A100   | 8                  | 4                | 60.9             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1junYUKA64swtwad9nbaoZu4XmBNIxokL?usp=sharing)                 |  [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1vaW6pay46K9XxLYSEpFls2qeoUB6au-1/view?usp=sharing)                |
| SegFormer-B5 RGB Norm ImageNet   | NVIDIA Tesla A100   | 8                  | 4                | 61.3             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1lDq9hhcJs7mtmbt3f4zY7NlSDlcfQDi5?usp=sharing)                 | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1djvqk9nXSqIc4dh_lwyKnAp9gZUQIMSf/view?usp=sharing)                 |
| SegFormer-B5 IGB Norm Aerial     | NVIDIA Tesla V100   | 4                  | 8                | 62.4             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1G03xSQaqoy5gk2hFouCcfZIIf8Sqbc65?usp=sharing) |[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/17Cwkb2vIiXiXxhZaZ-98Cuusne_mp4KX/view?usp=sharing)               |



## Inférence groupée & Pseudo labeling

Nous avons ensuite effectué une inférence simultanée des 4 modèles sur le jeu de test.

![Modèles ensemblistes](https://raw.githubusercontent.com/alanent/flair2_ign_2nd_place/main/assets/ensemble_models.png)

Le résultat est ensuite utilisé pour réentrainer le modèle suivant sur 2 époques supplémentaires.

| Modèle                                            | GPU d'Entraînement  | Lot d'Entraînement | Nombre d'Époques | Mean IoU (test)  | Models Link      | Colab Link       |
|---------------------------------------------------|---------------------|--------------------|------------------|------------------|------------------|------------------|
| SegFormer-B5 IGB Norm Aerial + Pseudo labeling    | NVIDIA Tesla V100   | 4                  | 2                | 63.3             | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/14QWUJzTqbJfjtE54587aJbLenawV-7Lm/view?usp=sharing)                   |


## Preprocessing des données sentinel2

Exécutez `./scripts/preprocess_sentinel2_files.py`.


## Conversion des Modèles en ONNX FP16

Exécutez `./scripts/convert_segformers_onnx.py`.


## Solution finale

Notre solution est l'inférence simultanée des modèles : `SegFormer-B5 IGB Norm Aerial pseudo labeled` + `SegFormer-B5 RGB Norm Sentinel2`. Une sélection des résultats par classe est effectué dans un deuxième temps. Le temps total d'inférence des modèles sous l'environnement Colab décrit ci-dessous est de 46 minutes et le score obtenu est de 63.55 IoU. 


### Environnement d'Exécution

- **Environnement**: Colab Pro
- **GPU**: NVIDIA Tesla T4
- **RAM**: Haute capacité

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1mM4oTfXj6wthzVneihG-BHHI800BOz_M/view?usp=sharing)

### Execution locale

1. **Installation des Dépendances**

`pip install -r requirements.txt`


2. **Télécharger les données et les modèles**

Télécharger et décompresser les données et les modèles respectivant dans les dossiers `/data` & `/models` 
- MODELES ONNX (float16) : https://drive.google.com/drive/folders/12ll_y0AaqEA9-EpajPM_OADoJrOuLc1c?usp=drive_link
- données de test (données sentinel2 préprocessées + images aeriennes à la racine du même dossier) : https://drive.google.com/drive/folders/1MJ_Cc4lRRQDEbmLvi0GeGwP__4DxH5-z?usp=sharing
  
3. **Inférence**

 exécutez `./scripts/run.py`.




