# nnProjectNanterre

Répertoire du projet de réseaux de neurones dans le cadre du Master 2 TAL à l'Université Paris Nanterre.

**----Travail en cours----**

**Objectif** : construire une app de TAL constituée d'une interface où l'utilisateur peut rentrer des coups d'échecs et recevoir un commentaire à chaque coup, qui sera généré automatiquement par un modèle BART pré-entraîné et fine-tunné par nos soins. Le modèle est capable de générer un commentaire à partir d'une notation FEN passée en entrée.

## BACKEND

**Ordre de lancement des scripts (pour le moment)** :

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
    Ce script télécharge localement le modèle BART "facebook/bart-large" et son Tokenizer.

4. **pre_processing.py**  
    Ce script contient :
    - Une fonction qui renvoie le chemin du dossier contenant les csv des matchs (find_corpus_folder) ;
    - Une fonction qui renvoie, pour une notation FEN en entrée, une notation FEN encodée (encode_fen) ;
    - Une fonction qui renvoie, pour un commentaire en entré, un commentaire encodée (encode_comment) ;
    - Une fonction qui renvoie, pour une notation UCI en entrée, une notation UCI encodée (encoded_uci).

5. **model_test_1.py = BASELINE**
    Ce script utilise le modèle BART pour générer un commentaire à partir d'une notation FEN donnée en entrée. C'est une version de base. La génération est de très mauvaise qualité et le modèle n'est pas entraîné sur les données contenues dans le dossier "corpus_csv".

6. **model_test_2.py = MODELE DE GENERATION DE COMMENTAIRE**
    Ce script est dédié au fine-tuning du modèle BART pour la génération de commentaires à partir de notations FEN d'échecs. Voici les étapes principales de ce script :
    - Chargement des données d'entraînement et de test préparées à partir des parties d'échecs annotées.
    - Entraînement du modèle BART sur les données d'entraînement pour adapter le modèle à la tâche de génération de commentaires.
    - Évaluation des performances du modèle fine-tuné sur les données de test.
    - Test du modèle fine-tuné en générant un commentaire pour une notation FEN donnée à titre d'exemple.
    Pour exécuter ce script, assurez-vous d'avoir les données préparées dans le répertoire "corpus_csv" et d'avoir exécuté les scripts nécessaires pour le téléchargement du modèle BART et son tokenizer, ainsi que pour les fonctions de prétraitement.

7. **model_test_3.py = MODELE POUR JOUER AUX ECHECS**
    Ce script est conçu pour le fine-tuning du modèle BART afin de générer des mouvements d'échecs en notation UCI à partir de positions d'échecs en notation FEN. Voici les étapes principales de ce script :
    - Téléchargement et initialisation du modèle BART ainsi que de son tokenizer.
    - Extraction des données d'entraînement et de test à partir des parties d'échecs annotées.
    - Fine-tuning du modèle BART sur les données d'entraînement pour adapter le modèle à la tâche de génération de mouvements d'échecs.
    - Évaluation des performances du modèle fine-tuné sur les données de test.
    - Prédiction du prochain mouvement d'échecs à partir d'une position donnée en notation FEN en utilisant le modèle fine-tuné.
    Pour exécuter ce script, assurez-vous d'avoir les données préparées dans le répertoire "corpus_csv" et d'avoir exécuté les scripts nécessaires pour le téléchargement du modèle BART et son tokenizer, ainsi que pour les fonctions de prétraitement.

8. **model_test_4.py = MODELE MULTITACHE (génère commentaire et joue aux échecs)**
    Ce script vise à l'entraînement d'un modèle BART en utilisant une approche multi-tâches pour la génération de commentaires sur les parties d'échecs et la prédiction des prochains mouvements en notation UCI. Voici les principales étapes de ce script :
    - Téléchargement et initialisation du modèle BART ainsi que de son tokenizer.
    - Extraction des données nécessaires pour le fine-tuning, y compris les vocabulaires pour les différentes tâches.
    - Fine-tuning du modèle BART sur les données d'entraînement pour les deux tâches : génération de commentaires et prédiction UCI.
    - Utilisation d'une approche multi-tâches pour entraîner le modèle, combinant les données d'entraînement pour les deux tâches.
    - Évaluation des performances du modèle fine-tuné.
    - Fonction pour générer à la fois un commentaire et un mouvement à partir d'une position d'échecs donnée.
    Pour exécuter ce script, assurez-vous d'avoir les données préparées dans le répertoire "corpus_csv" et d'avoir exécuté les scripts nécessaires pour le téléchargement du modèle BART et son tokenizer, ainsi que pour les fonctions de prétraitement et les scripts `model_test_2.py` et `model_test_3.py` pour préparer les données et les loaders d'entraînement.

**WARNING**

    En raison des performances limitées du GPU à notre disposition, l'entraînement du modèle 4 (multitâche) a été effectué avec les spécifications suivantes :

    - TrainSet = Environ 16 900 exemples (1 exemple = 1 mouvement commenté + UCI + FEN)
    - TestSet = Environ 3 900 exemples
    - Taille du lot (Batch_size) = 4
    - Temps d'entraînement : environ 3 jours sur nos machines personnelles

## FRONTEND ET API