# Liver-cancer-detection
Analysis of a cohort of patients with liver cancer

## The data

- Les numéros de patients sont réindexés pour chaque classe (il y a trois numéros un pour CHC, CCK et mixte)

- Tous les patients ne possèdent pas les quatre temps pour l’IRM (ça me semble peu pertinent de faire de l’imputation de données manquantes au vue du nombre de données)

- Pour la récidive, je n’ai pas encore regardé le nombre de patients mais c’est peut-être très faible

- Quand on travaille sur les données radiomiques générales (à l’échelle du patient), ça peut être intéressant de regarder la slice du milieu pour comparer (a priori c’est là ou il y aura le plus de tumeur donc les features seraient plus intéressantes)

- Pour la classification de classes de cancers, on nous a recommandé en projet de commencer par CHC vs CCK et après introduire les mixtes

- Les colonnes commençant par « diagnostics » sont inutiles de mémoire




### Description des Colonnes dans `Descriptif_patients.xlsx`

- **ID** : Identifiant unique pour chaque patient
- **classe_name** : Nom de la classe ou du type de cancer du foie, utilisé pour identifier un groupe spécifique de patients. 87 carcinomes hépatocellulaires (`CCK`), 23 cholangiocarcinomes (`CHC`), 37 tumeurs mixtes (`Mixtes`). 
- **Gender** : Sexe du patient, indiqué par "M" pour masculin et "F" pour féminin.
- **Age_at_disease** : Âge du patient au moment du diagnostic de la maladie du foie. Exemples: 62 ans, 57 ans, etc.
- **Date_of_MRI** : Date à laquelle l'IRM (Imagerie par Résonance Magnétique) a été réalisée.
- **Date_of_surgery** : Date de l'opération chirurgicale, si elle a eu lieu.
- **Alpha_foetoprotein** : Niveau d'alpha-fœtoprotéine, un biomarqueur souvent utilisé pour le diagnostic des carcinomes hépatocellulaires (CCK)
- **Local_relapse** : Indique si une récidive locale du cancer a été enregistrée (1.0 pour oui, 0.0 pour non).
- **Date_of_local_relapse** : Date de la récidive locale si elle a eu lieu. 
- **Distant_relapse** : Indique si le cancer a métastasé à d'autres parties du corps (1.0 pour oui, 0.0 pour non).
- **Date_of_distant_relapse** : Date à laquelle une métastase à distance a été détectée, si applicable.
- **Death** : Indique si le patient est décédé (1.0 pour oui, 0.0 pour non).
- **Date_of_death** : Date de décès du patient, si applicable. 
- **Date_of_lost_of_FU** : Dernière date à laquelle le patient a été suivi avant de perdre le contact.




### Description des Caractéristiques de `radiomiques_global.xlsx`

#### Informations de Base
- **ID** : Identifiant unique pour chaque patient
- **classe_name** : Type ou classe de cancer du foie.
- **temps_inj** : Temps après l'injection d'un agent de contraste. Les valeurs possibles incluent :
  - **VEIN** : Phase veineuse
  - **TARD** : Phase tardive
  - **PORT** : Phase portale
  - **ART** : Phase artérielle
- **patient_num** : Identifiant unique pour chaque patient. Chaque numéro peut apparaître jusqu'à quatre fois pour chaque valeur de `temps_inj`.

#### Caractéristiques Statistiques de Premier Ordre
Les caractéristiques suivantes décrivent la distribution des intensités des pixels au sein de la région définie par le masque de la tumeur :
- **original_firstorder_10Percentile** : 10e percentile de la distribution d'intensité.
- **original_firstorder_90Percentile** : 90e percentile de la distribution d'intensité.
- **original_firstorder_Energy** : Somme des valeurs d'intensité au carré.
- **original_firstorder_Entropy** : Mesure de la randomicité ou complexité de la texture de l'image.
- **original_firstorder_InterquartileRange** : Différence entre le 75e et le 25e percentile des valeurs d'intensité.
- **original_firstorder_Kurtosis** : Mesure de l'aplatissement de la distribution d'intensité.
- **original_firstorder_Maximum** : Valeur maximale d'intensité.
- **original_firstorder_Minimum** : Valeur minimale d'intensité.
- **original_firstorder_Mean** : Moyenne des valeurs d'intensité.
- **original_firstorder_Median** : Médiane des valeurs d'intensité.
- **original_firstorder_MeanAbsoluteDeviation** : Moyenne des écarts absolus par rapport à la moyenne d'intensité.
- **original_firstorder_Range** : Différence entre les valeurs maximale et minimale d'intensité.
- **original_firstorder_RobustMeanAbsoluteDeviation** : Écart moyen absolu, calculé entre les 10e et 90e percentiles.
- **original_firstorder_RootMeanSquared** : Racine carrée de la moyenne des valeurs d'intensité au carré.
- **original_firstorder_Skewness** : Mesure de l'asymétrie de la distribution d'intensité.
- **original_firstorder_TotalEnergy** : Somme totale des valeurs d'intensité au carré, mise à l'échelle par le volume du voxel.
- **original_firstorder_Uniformity** : Uniformité de la distribution d'intensité.
- **original_firstorder_Variance** : Variance des valeurs d'intensité.

#### Caractéristiques Basées sur la Forme
Les caractéristiques de la forme de la région tumorale incluent :
- **original_shape_Elongation** : Élongation de la forme de la tumeur.
- **original_shape_Flatness** : Platitude de la forme de la tumeur.
- **original_shape_LeastAxisLength** : Longueur de l'axe le moins étendu.
- **original_shape_MajorAxisLength** : Longueur de l'axe majeur.
- **original_shape_Maximum2DDiameterColumn, Row, Slice** : Diamètres maximaux dans les projections 2D.
- **original_shape_Maximum3DDiameter** : Diamètre maximal à travers le volume.
- **original_shape_MeshVolume** : Volume de la maille.
- **original_shape_MinorAxisLength** : Longueur de l'axe mineur.
- **original_shape_Sphericity** : Sphéricité de la forme de la tumeur.
- **original_shape_SurfaceArea** : Surface de la tumeur.
- **original_shape_SurfaceVolumeRatio** : Rapport surface/volume.
- **original_shape_VoxelVolume** : Volume des voxels.

#### Caractéristiques de Texture
Descripteurs de texture extraits de l'image, incluant plusieurs mesures pour chaque type de texture :
- **original_glcm_[Caractéristique]** : Caractéristiques de la Matrice de Cooccurrence des Niveaux de Gris (GLCM).
- **original_gldm_[Caractéristique]** : Caractéristiques de la Matrice de Dépendance des Niveaux de Gris (GLDM).
- **original_glrlm_[Caractéristique]** : Caractéristiques de la Matrice de Longueur de Séquence des Niveaux de Gris (GLRLM).
- **original_glszm_[Caractéristique]** : Caractéristiques de la Matrice de Zone de Taille des Niveaux de Gris (GLSZM).
- **original_ngtdm_[Caractéristique]** : Caractéristiques de la Matrice de Différence de Tons Voisins des Niveaux de Gris (NGTDM).

### Description des Colonnes de `radiomiques_multislice.xlsx`

#### Informations Générales
- **ID** : Identifiant unique pour chaque patient
- **slice_num** : Numéro de la coupe transversale de l'imagerie pour laquelle les caractéristiques ont été extraites.
- **classe_name** : Type ou classe de cancer du foie.
- **temps_inj** : Phase d'imagerie post-injection de l'agent de contraste, avec les valeurs possibles :
  - **VEIN** : Phase veineuse
  - **TARD** : Phase tardive
  - **PORT** : Phase portale
  - **ART** : Phase artérielle

#### Same as `radiomiques_global.xlsx` for the rest

