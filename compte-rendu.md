---
title: Concevoir, analyser et comparer des modèles d'apprentissage automatique.
author: ANTOINE Marcoly & DIALLO Abdoul Aziz
date: 20/04/2024
...

# 1 - Préparation des données

1. Ces données comporte 15 attributs au total comme le montre `data.shape[1]`.

   - Les en-têtes de colonnes de la première ligne du fichier listent 14 attributs de Attr_A à Attr_N et une colonne supplémentaire appelée Class.
   - Chaque ligne de données contient 15 valeurs, une pour chaque attribut et une dernière pour la classe de l'enregistrement.

   Ainsi, le fichier synthetic.csv comporte 14 attributs (Attr_A à Attr_N) et une classe (Class), ce qui donne un total de 15 attributs au total.

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
