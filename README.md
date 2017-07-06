# 3DLearning

Ce dépôt contient le code utilisé pour la reconnaissance de modèle 3D via utilisation de réseaux de neurones. Réalisé par Etienne Houzé lors d'un stage à Acute3D effectué de mars a juillet 2017.

## Prérequis
Le code contient une partie en C++, une autre en Python.
### C++
Le code C++ a été rédigé par... Il nécessite d'être compilé avec Visual Studio 2015 64bit, version Release. Toutes les librairies requises sont présentes dans le dossier des sources.
###
Le code Python a été rédigé au cours du stage. Il exploite l'API Keras en version 2.0, elle-même basée sur TensorFlow 1.1. Ces deux packages sont donc bien évidemment nécessaires, ainsi que toutes leurs dépendances.
Sont aussi requis les packages numpy, pandas, pillow, cython

## How to use
- Le dossier Python/scripts contient différents scripts Python qui exécutent les différentes tâches du processus d'apprentissage et d'évaluation d'un modèle.
- Le dossier C++ contient différents scripts qui permettent de générer un dataset à partir d'un mesh donné sous format .obj.

### Génération du dataset d'images
1. A partir d'un projet ContextCapture, il faut effectuer deux reconstructions au format obj.
2. Sur l'une de ces deux reconstructions, on peint les étiquettes à l'aide d'un logiciel 3D, type Blender.
3. A l'aide du script Python camera_generation.py, on génère les positions de caméras qui serviront pour l'entrainement et pour la validation. On pourra renommer les fichiers d'export "val_cam.xml" et "train_cam.xml"
3. On applique le script C++ TextureRender quatres fois, une fois sur "val_cam.xml", en utilisant l'obj correspondant aux textures RGB puis sur l'obj correspondant aux textures étiqutées. On refait de même avec "train_cam.xml".
4. Sur chaque fichier d'export, on utilise les scripts DepthRender, AltRender produisant les maps de profondeur et d'altitude, ainsi que le fichier de projection.
5. On crée deux dossiers, un de validation, un d'entrainenemt.
6. Dans chaque, on crée les dossiers "Altitude", "Depth", "RGB", "RGB_labels" ainsi que le fichier "Projections.txt" correspondants.
7. On utilise le script Python image_preprocess afin de crée un dossier "Labels" correspondant aux étiquettes.

**Attention : Il est impératif que le dossier d'apprentissage, comme le dossier de validation, comprenne les sous-répertoires suivants : RGB, Labels, Depth, Altitude, ainsi qu'un fichier txt comprenant la correspondane couleur/labeln nommé "labels.txt" et le fichier Projections.txt**

### Apprentissage sur le dataset d'images

- Le script permettant d'entrainer un réseau de neurone est scripts/training3D.py. Se référer à la documentation attachée à ce script pour son utilisation.
- A l'issue de ce script, le modèle du réseau sera créé puis sauvegardé dans le répertoire mentionné au script précédent. On pourra alors soit continuer l'entrainenement, en chargeant les poids, ou alors l'évaluer, avec le scripts evaluate3D.py
- Enfin, on peut directement l'utiliser sur un set d'images afin de les étiqueter. On utilisera pour ceci le script "inférence3D.py"
- Chaque modèle est prévu pour être stocké dans un dossier qui contiendra le fichier de sauvegarde des poids, le fichier de configuration, ainsi qu'un schéma du réseau généré par Keras. Ces fichiers ne sont pas destinés à être modifié par l'utilisateur. Aussi, il est recommandé de créer un nouveau répertoire par modèle, même en cas de reprise d'entrainement.
