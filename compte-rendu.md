---
title: Concevoir, analyser et comparer des modèles d'apprentissage automatique.
author: ANTOINE Marcoly & DIALLO Abdoul Aziz
date: 20/04/2024
...

# 1 - Préparation des données

1. Ces données comporte 14 attributs au total comme le montre `data.shape[1]`.

   - Les en-têtes de colonnes de la première ligne du fichier listent 14 attributs de Attr_A à Attr_N et un label Class.

2. Le nombre de classe différente dans les données est : 4

3. Voici une statistique simplifiée de nos données :

   | Classe | Nombre d'instances |
   | :----: | :----------------: |
   |   0    |        674         |
   |   1    |        908         |
   |   2    |        472         |
   |   3    |        244         |

4. Non, si on observe le schéma 1 on voit que les données ne le sont pas.
   De plus si l'on choisit de les ranger par classe , on peut voir clairement que ce n'est pas divisible linéairement à l'état brut.

   ![Non lineaire independant](./lineairement_independant.png "Schema 1")

5. - Pour l'arbre de décision
     Puisque les variables sont déjà numériques et que la colonne Class est utilisée comme étiquette (et non comme une fonctionnalité), aucun encodage One-hot n'est nécessaire pour les fonctionnalités. Si Class était utilisée comme une caractéristique d'entrée plutôt que comme une étiquette, et si elle comprenait de nombreuses catégories différentes, l'encodage One-hot pourrait être envisagé pour éviter de donner un ordre artificiel entre les catégories.
     Pour un modèle basé sur un arbre de décision, la normalisation des données n'est généralement pas nécessaire. Les arbres de décision ne sont pas sensibles à la magnitude des valeurs des attributs de la même manière que le sont les modèles basés sur des calculs de distance ou des modèles linéaires. Voici pourquoi :

   - Pour les réseaux de neurones : l'encodage one-hot est nécéssaire de manière à dissoier les sorties de classe

6. Séparer les données en jeu d'entraînement et jeu de test permet d'évaluer la performance du modèle de manière impartiale, de détecter le surapprentissage et d'obtenir des prédictions plus précises sur des données nouvelles.

# Mise en oeuvre des modèles

Les _quartiles_ sont des valeurs qui divisent les données en quatre parties égales lorsqu'elles sont triées dans l'ordre croissant. Plus précisément, les quartiles sont les valeurs situées à 25 %, 50 % et 75 % de l'ensemble de données ordonné. Voici les trois quartiles couramment utilisés :

- **Premier quartile (Q1)** : également appelé le quartile inférieur, il est la valeur qui se situe au 25 % inférieur des données ordonnées.
- **Deuxième quartile (Q2)** : également appelé la médiane, c'est la valeur qui divise les données en deux moitiés égales (50 %).
- **Troisième quartile (Q3)** : également appelé le quartile supérieur, il est la valeur qui se situe au 75 % des données ordonnées.

Les quartiles sont des mesures de tendance centrale qui fournissent des informations sur la dispersion et la répartition des données.

Pour les calculer nous avons utilisé la méthode `quantile()` sur la série des données d'un attribut spécifique. Pour donner un exemple sur l'attribut A.

```python
attribute = 'Attr_A'

# Calculer les quartiles pour l'attribut choisi
quartiles = data[attribute].quantile([0.25, 0.5, 0.75])

# Afficher les quartiles
print(f"Quartile 1 (Q1) de l'attribut '{attribute}': {quartiles[0.25]}")
print(f"Médiane (Q2) de l'attribut '{attribute}': {quartiles[0.5]}")
print(f"Quartile 3 (Q3) de l'attribut '{attribute}': {quartiles[0.75]}")
```

# 3. Analyse des modèles

Nous avons fait le choix d'utiliser les modèles donées en fichier de test afin de répondre à cette question.

| Modèle                | Accuracy | Précision C1 | Rappel C1 | F1-score C1 | Précision C2 | Rappel C2 | F1-score C2 | Précision C3 | Rappel C3 | F1-score C3 | Précision C4 | Rappel C4 | F1-score C4 |
| --------------------- | -------- | ------------ | --------- | ----------- | ------------ | --------- | ----------- | ------------ | --------- | ----------- | ------------ | --------- | ----------- |
| y_pred_DT4            | 0.3399   | 0.3399       | 1.0000    | 0.5073      | 0.0000       | 0.0000    | 0.0000      | 0.0000       | 0.0000    | 0.0000      | 0.0000       | 0.0000    | 0.0000      |
| y_pred_DT5            | 0.3399   | 0.3399       | 1.0000    | 0.5073      | 0.0000       | 0.0000    | 0.0000      | 0.0000       | 0.0000    | 0.0000      | 0.0000       | 0.0000    | 0.0000      |
| y_pred_DT6            | 0.3399   | 0.3399       | 1.0000    | 0.5073      | 0.0000       | 0.0000    | 0.0000      | 0.0000       | 0.0000    | 0.0000      | 0.0000       | 0.0000    | 0.0000      |
| y_pred_NN_relu_10-8-4 | 0.8824   | 0.9714       | 0.8718    | 0.9189      | 0.8717       | 0.9819    | 0.9235      | 0.8351       | 0.9101    | 0.8710      | 0.7143       | 0.5208    | 0.6024      |
| y_pred_NN_relu_10-8-6 | 0.8736   | 0.9712       | 0.8654    | 0.9153      | 0.9012       | 0.9337    | 0.9172      | 0.8081       | 0.8989    | 0.8511      | 0.6327       | 0.6458    | 0.6392      |
| y_pred_NN_relu_6-4    | 0.8693   | 0.9379       | 0.8718    | 0.9037      | 0.9075       | 0.9458    | 0.9263      | 0.7885       | 0.9213    | 0.8497      | 0.6486       | 0.5000    | 0.5647      |
| y_pred_NN_tanh_10-8-4 | 0.8475   | 0.9045       | 0.9103    | 0.9073      | 0.8670       | 0.9819    | 0.9209      | 0.7368       | 0.9438    | 0.8276      | 0.0000       | 0.0000    | 0.0000      |
| y_pred_NN_tanh_10-8-6 | 0.9107   | 0.9664       | 0.9231    | 0.9443      | 0.8950       | 0.9759    | 0.9337      | 0.9101       | 0.9101    | 0.9101      | 0.7750       | 0.6458    | 0.7045      |
| y_pred_NN_tanh_6-4    | 0.8344   | 0.8476       | 0.8910    | 0.8688      | 0.8684       | 0.9940    | 0.9270      | 0.7524       | 0.8876    | 0.8144      | 0.0000       | 0.0000    | 0.0000      |

### y_pred_DT4

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 130 | 19  | 6   | 1   |
| 1                            | 15  | 147 | 2   | 2   |
| 2                            | 25  | 20  | 44  | 0   |
| 3                            | 12  | 8   | 22  | 6   |

### y_pred_DT5

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 126 | 14  | 10  | 6   |
| 1                            | 12  | 140 | 9   | 5   |
| 2                            | 12  | 5   | 67  | 5   |
| 3                            | 7   | 5   | 15  | 21  |

### y_pred_DT6

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 130 | 14  | 10  | 2   |
| 1                            | 9   | 147 | 3   | 7   |
| 2                            | 9   | 7   | 71  | 2   |
| 3                            | 5   | 7   | 11  | 25  |

### y_pred_NN_relu_10-8-4

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 136 | 8   | 5   | 7   |
| 1                            | 0   | 163 | 2   | 1   |
| 2                            | 2   | 4   | 81  | 2   |
| 3                            | 2   | 12  | 9   | 25  |

### y_pred_NN_relu_10-8-6

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 135 | 9   | 5   | 7   |
| 1                            | 0   | 155 | 4   | 7   |
| 2                            | 2   | 3   | 80  | 4   |
| 3                            | 2   | 5   | 10  | 31  |

### y_pred_NN_relu_6-4

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 136 | 6   | 5   | 9   |
| 1                            | 3   | 157 | 5   | 1   |
| 2                            | 2   | 2   | 82  | 3   |
| 3                            | 4   | 8   | 12  | 24  |

### y_pred_NN_tanh_10-8-4

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 142 | 9   | 5   | 0   |
| 1                            | 0   | 163 | 3   | 0   |
| 2                            | 2   | 3   | 84  | 0   |
| 3                            | 13  | 13  | 22  | 0   |

### y_pred_NN_tanh_10-8-6

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 144 | 5   | 1   | 6   |
| 1                            | 0   | 162 | 2   | 2   |
| 2                            | 3   | 4   | 81  | 1   |
| 3                            | 2   | 10  | 5   | 31  |

### y_pred_NN_tanh_6-4

| True label \ Predicted label | 0   | 1   | 2   | 3   |
| ---------------------------- | --- | --- | --- | --- |
| 0                            | 139 | 9   | 8   | 0   |
| 1                            | 1   | 165 | 0   | 0   |
| 2                            | 4   | 6   | 79  | 0   |
| 3                            | 20  | 10  | 18  | 0   |

# 4. Le meilleur modèle

L'analyse comparative des performances de nos divers modèles révèle que certains réseaux de neurones, notamment les modèles `y_pred_NN_relu_10-8-6` et `y_pred_NN_tanh_10-8-6`, surpassent les arbres de décision (`y_pred_DT4`, `y_pred_DT5`, `y_pred_DT6`) en termes de précision, de rappel, et de score F1. Cette supériorité indique que les réseaux de neurones sont plus aptes à gérer les complexités de notre ensemble de données, qui pourrait intégrer des éléments sophistiqués tels que des diagnostics médicaux ou des anomalies dans des systèmes critiques.

### Raisons de la préférence pour un type de modèle

- **Exactitude et spécificité :** Les modèles de réseaux de neurones, en particulier `y_pred_NN_relu_10-8-6`, ont démontré une capacité supérieure à classer précisément les différentes catégories. Cette caractéristique est essentielle dans des domaines où les erreurs de classification peuvent entraîner des conséquences significatives, comme c'est le cas en diagnostic médical.
- **Interprétabilité et compréhension :** Malgré la réputation de "boîte noire" des réseaux de neurones, des méthodes de visualisation et d'explication des décisions modélisées peuvent permettre de mieux comprendre ces modèles. Une communication claire des résultats du modèle peut également rendre les décisions plus transparentes pour toutes les parties prenantes, y compris les patients dans un contexte médical.

Ces considérations mettent en lumière l'importance de sélectionner un modèle non seulement pour ses performances techniques, mais aussi pour sa pertinence dans le contexte spécifique de son application.
