# nnProjectNanterre

Répertoire du projet de réseaux de neurones dans le cadre du Master 2 TAL à l'Université Paris Nanterre.

**Objectif** : construire une app de TAL constituée d'une interface où l'utilisateur peut rentrer des coups d'échecs et recevoir un commentaire à chaque coup, qui sera généré automatiquement par un modèle BART pré-entraîné et fine-tunné sur nos data. Le modèle sera capable de générer un commentaire à partir d'une notation FEN passée en entrée.

## BACKEND

**Ordre de lancement des scripts** :

### MODELE 1 : BART

1. **links_collection.py**  
    Ce script parcourt le fichier "saved_links.p" qui contient les liens des parties annotées présentes dans la base de données de gameknot.com. En sortie, ce script donne une liste des liens présents dans le fichier.

2. **scraping_links.py**  
    Ce script effectue un scraping sur les liens obtenus avec "links_collection.py". Il produit en sortie un dossier "corpus_csv" contenant un fichier CSV par partie, nommés progressivement "df_match_1", "df_match_2", etc. Chaque fichier CSV contient les colonnes suivantes :
    - **N_move** (1, 2, 3,...) : indexation de chaque coup ;
    - **Standard_notation** (e4, f5,...) : notation standard qui décrit la case où la pièce a été déplacée ;
    - **UCI_notation** (e2e4, f7f5,...) : notation UCI qui décrit la case où la pièce a été déplacée + la case de départ ;
    - **FEN_notation** (rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1) : décrit l'état de l'échiquier après le coup ;
    - **Comment** ("jusqu'à présent, tout va bien...") : commentaire relatif au coup.

3. **download_BART_model.py**  
    Ce script télécharge depuis la librairie Transformers de HF le modèle BART "facebook/bart-large" et son Tokenizer, et le modèle MBartForConditionalGeneration et son tokenizer MBart50TokenizerFast.

4. **pre_processing.py**  
    Ce script contient :
    - Une fonction qui renvoie le chemin du dossier contenant les csv des matchs (find_corpus_folder) ;
    - Une fonction qui renvoie, pour une notation FEN en entrée, une notation FEN encodée (encode_fen) avec le tokenizer de BART ;
    - Une fonction qui renvoie, pour un commentaire en entré, un commentaire encodée (encode_comment) avec le tokenizer de BART ;
    - Une fonction qui renvoie, pour une notation UCI en entrée, une notation UCI encodée (encoded_uci) avec le tokenizer de BART.

5. **model_bart_1.py = BASELINE**
    Ce script utilise le modèle BART pour générer un commentaire à partir d'une notation FEN donnée en entrée. C'est une version de base. La génération est de très mauvaise qualité et le modèle n'est pas entraîné sur les données contenues dans le dossier "corpus_csv".

6. **model_bart_2.py = MODELE DE GENERATION DE COMMENTAIRE**
    Ce script est dédié au fine-tuning du modèle BART pour la génération de commentaires à partir de notations FEN d'échecs. Voici les étapes principales de ce script :
    - Chargement des données d'entraînement et de test préparées à partir des parties d'échecs annotées.
    - Entraînement du modèle BART sur les données d'entraînement pour adapter le modèle à la tâche de génération de commentaires.
    - Évaluation des performances du modèle fine-tuné sur les données de test.
    - Test du modèle fine-tuné en générant un commentaire pour une notation FEN donnée à titre d'exemple.
    Pour exécuter ce script, assurez-vous d'avoir les données préparées dans le répertoire "corpus_csv" et d'avoir exécuté les scripts nécessaires pour le téléchargement du modèle BART et son tokenizer, ainsi que pour les fonctions de prétraitement.

7. **model_bart_3.py = MODELE POUR JOUER AUX ECHECS**
    Ce script est conçu pour le fine-tuning du modèle BART afin de générer des mouvements d'échecs en notation UCI à partir de positions d'échecs en notation FEN. Voici les étapes principales de ce script :
    - Téléchargement et initialisation du modèle BART ainsi que de son tokenizer.
    - Extraction des données d'entraînement et de test à partir des parties d'échecs annotées.
    - Fine-tuning du modèle BART sur les données d'entraînement pour adapter le modèle à la tâche de génération de mouvements d'échecs.
    - Évaluation des performances du modèle fine-tuné sur les données de test.
    - Prédiction du prochain mouvement d'échecs à partir d'une position donnée en notation FEN en utilisant le modèle fine-tuné.
    Pour exécuter ce script, assurez-vous d'avoir les données préparées dans le répertoire "corpus_csv" et d'avoir exécuté les scripts nécessaires pour le téléchargement du modèle BART et son tokenizer, ainsi que pour les fonctions de prétraitement.

8. **model_bart_4.py = MODELE MULTITACHE (génère commentaire et joue aux échecs)**
    Ce script vise à l'entraînement d'un modèle BART en utilisant une approche multi-tâches pour la génération de commentaires sur les parties d'échecs et la prédiction des prochains mouvements en notation UCI. Voici les principales étapes de ce script :
    - Téléchargement et initialisation du modèle BART ainsi que de son tokenizer.
    - Extraction des données nécessaires pour le fine-tuning, y compris les vocabulaires pour les différentes tâches.
    - Fine-tuning du modèle BART sur les données d'entraînement pour les deux tâches : génération de commentaires et prédiction UCI.
    - Utilisation d'une approche multi-tâches pour entraîner le modèle, combinant les données d'entraînement pour les deux tâches.
    - Évaluation des performances du modèle fine-tuné.
    - Fonction pour générer à la fois un commentaire et un mouvement à partir d'une position d'échecs donnée.
    Pour exécuter ce script, assurez-vous d'avoir les données préparées dans le répertoire "corpus_csv" et d'avoir exécuté les scripts nécessaires pour le téléchargement du modèle BART et son tokenizer, ainsi que pour les fonctions de prétraitement et les scripts `model_bart_2.py` et `model_bart_3.py` pour préparer les données et les loaders d'entraînement.

### MODELE 2 : MBART

9. **model_mbart.py = MODELE DE GENERATION DE COMMENTAIRE (approche avec un modèle de traduction automatique)**
   Ce script vise à l'entraînement d'un modèle MBART pour la génération de commentaires sur les parties d'échecs. Voici les principales étapes de ce script :
    - Téléchargement et initialisation du modèle MBART ainsi que de son tokenizer.
    - une fonction de génération de commentaire avec le modèle de base (baseline) -> le commentaire généré ressemble à une fen.
    - Extraction des données nécessaires pour le fine-tuning, y compris les vocabulaires pour les différentes tâches.
    - Fine-tuning du modèle MBART sur les données d'entraînement pour la génération de commentaires.
    - Évaluation des performances du modèle fine-tuné.
    - Fonction pour générer un commentaire à partir d'une position d'échecs donnée.
    La langue source choisie est le chinois ("Zh_CN") car elle est segmentée caractère par caractère, ce qui correspond à notre tokenization de la FEN. La langue cible est l'anglais.


**WARNING**

En raison des performances limitées du GPU à notre disposition, l'entraînement du modèle 4 (multitâche) a été effectué avec les spécifications suivantes :

- TrainSet = Environ 16 900 exemples (1 exemple = 1 mouvement commenté + UCI + FEN)
- TestSet = Environ 3 900 exemples
- Taille du lot (Batch_size) = 4
- Temps d'entraînement : environ 3 jours sur nos machines personnelles

Les résultats très peu satisfaisant (génération de ponctuation uniquement) du modèle 4 nous ont amené à essayer un autre modèle : MBart50, pré-entraîné à la traduction, et non à la summerization.

Nous avons donc ré-entraîné un modèle Mbart50 sur nos données, malheureusement, une fois encore, la puissance de nos machines à été très limitante. Nous avons pu entraîné au max 12h, sur une batch_size de 1, donc sur (trop) peu d'exemples.

C'est pourquoi ce modèle génère des commentaires qui n'ont aucun sens, mais au moins il génère des mots ! Nous considérons donc ceci comme une réussite.

## FRONTEND ET API

HTML, CSS, JS.  

Pour le design graphique, nous avons opté pour l'utilisation d'une template qui nous plaisait : "Gallerised", de OS Templates; et de l'utilisation de Bootstrap.

Pour avoir un board d'échec, il a fallu recourir à la librairie js "chessboard.js" (chessboard.js v1.0.0 | (c) 2019 Chris Oakman | MIT License chessboardjs.com/license). Comme la doc de cette librairie conseillait d'installer chessboard via npm, il a fallu installer node et npm sur nos ordinateurs, et faire de ce projet un projet node. Mais, faute de complications et d'allourdissement inutiles, nous avons abandonné cette idée.

Chessboard permet l'affichage d'un board d'échec, mais il n'y a pas la logique des échecs intégrée à cette librairie. Pour cela, il a fallu installer une autre librairie js : chess.js (Copyright (c) 2023, Jeff Hlywa (jhlywa@gmail.com)).

Cette librairie nous a posé des difficultés à utiliser car elle est écrite en TypeScript. Il a donc fallu la convertir en js (un essai à été fait de la transpiler/compiler grâce à Babel, puis Webpack).
Cette librairie permet de jouer aux échecs sur le board de notre site selon les règles (seuls les coups légaux seront acceptés par exemple).

## Partie server : fast_api

La connexion entre le modèle et le frontend est faite via un server créé avec fastAPI, et grâce à Jinja pour le rendu du frontend.

## SOURCES

- Jhamtani, H., Gangal, V., Hovy, E., Neubig, G., & Berg-Kirkpatrick, T. 2018. Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data. Language Technologies Institute, Carnegie Mellon University.
- Swingle, C., & Mellsop, H. 2021. ChePT-2: Advancing the Application of Deep Neural Transformer Models to Chess Move Prediction and Self-Commentary. Département d'Informatique, Université Stanford. [Document PDF].