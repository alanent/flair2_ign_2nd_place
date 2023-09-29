# SegFormer Models Benchmark

Ce référentiel contient les résultats des tests de performance de différents modèles SegFormer sur diverses configurations matérielles et d'entraînement.
Source des scripts huggingface : https://huggingface.co/blog/fine-tune-segformer

Vous pouvez accéder à l'ensemble des données [ici](https://ignf.github.io/FLAIR/#FLAIR2).

## Entraînement

La solution proposée est d'utilisée différents type de preprocessing des données afin a créer de la variété dans nos données. Nous avons entrainé 4 segformers avec différents processing :

- Pour l'utilisation des données sentinels-2, nous avons filtrer les données contenant des nuages et de la neige. Nous enregistrons ensuite les données correpondantes à chaque image aérienne dans un fichier appelé SEN_{Img_id}.npy Pour chaque image aérienne, nous normalisons cette image par la moyenne et l'écart type de ce fichier .npy

- Pour la normalisation imagenet, nous avons repris le preprocessing effectué par la classe SegformerFeatureExtractor de HuggingFace.

- Pour la normalisation des images aeriennes nous utilisons, les moyennes et les écart-type de chaque couche sur l'ensemble du dataset.


| Modèle                           | GPU d'Entraînement  | Lot d'Entraînement | Nombre d'Époques | Mean IoU (test)  | Models Link      | Training Script  |
|----------------------------------|---------------------|--------------------|------------------|------------------|------------------|------------------|
| SegFormer-B5 RGB Norm Sentinel2  | NVIDIA Tesla A100   | 8                  | 4                | 61.3             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1GfI5OwrInzdMz--_VC0jY5AU13bxRJoy?usp=sharing)                 | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Cwkb2vIiXiXxhZaZ-98Cuusne_mp4KX?authuser=1#scrollTo=LDZvoduQLNjI)                |
| SegFormer-B5 RGB Norm Aerial     | NVIDIA Tesla A100   | 8                  | 4                | 60.9             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1junYUKA64swtwad9nbaoZu4XmBNIxokL?usp=sharing)                 | -                |
| SegFormer-B5 RGB Norm ImageNet   | NVIDIA Tesla A100   | 8                  | 4                | 61.3             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1lDq9hhcJs7mtmbt3f4zY7NlSDlcfQDi5?usp=sharing)                 | -                |
| SegFormer-B5 IGB Norm Aerial     | NVIDIA Tesla V100   | 4                  | 8                | 62.4             | [![Accéder aux modèles](https://img.shields.io/badge/Mod%C3%A8les-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/1G03xSQaqoy5gk2hFouCcfZIIf8Sqbc65?usp=sharing) |[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/17Cwkb2vIiXiXxhZaZ-98Cuusne_mp4KX/view?usp=sharing)               |

## Inférence groupée & Pseudo labeling

Inférence moyennée des 4 modèles 

![Modèles ensemblistes](https://raw.githubusercontent.com/alanent/flair2_ign_2nd_place/main/assets/ensemble_models.png)




## Pseudo labeling

Ce résultat est utilisé pour réentrainer le modèle suivant sur 2 époques supplémentaires.

| Modèle                                            | GPU d'Entraînement  | Lot d'Entraînement | Nombre d'Époques | Mean IoU (test)  | Colab Link       |
|---------------------------------------------------|---------------------|--------------------|------------------|------------------|------------------|
| SegFormer-B5 IGB Norm Aerial Pseudo labeling      | NVIDIA Tesla V100   | 4                  | 2                | 63.3             | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/14QWUJzTqbJfjtE54587aJbLenawV-7Lm/view?usp=sharing)                   |

## Environnement d'Exécution

- **Environnement**: Colab Pro
- **GPU**: NVIDIA Tesla T4
- **RAM**: Haute capacité

## Instructions d'Utilisation

1. **Installation des Dépendances**

pip install -r requirements.txt



2. **Télécharger les Données**

Vous pouvez accéder aux données [ici](https://ignf.github.io/FLAIR/#FLAIR2).




4. **Conversion des Modèles en ONNX FP16**

Exécutez `convert_models.py`.

Liens vers les poids des modèles ONNX (voir le tableau dans le notebook).

5. **Prédiction Finale**

SegFormer-B5 IGB Norm Aerial pseudo labeled + SegFormer-B5 RGB Norm Sentinel2 + sélection des classes = 63.55 IoU.

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1mM4oTfXj6wthzVneihG-BHHI800BOz_M/view?usp=sharing) ou exécutez `run.py`.




