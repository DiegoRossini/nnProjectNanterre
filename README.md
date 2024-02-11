# nnProjectNanterre

Répertoire du projet de réseaux de neurones dans le cadre du Master 2 TAL à l'Université Paris Nanterre.

**----Travail en cours----**

**Objectif actuel** : construire un modèle capable de générer un commentaire pertinent à partir d'un mouvement d'échecs en notation FEN passé en entrée.

**Ordre de lancement des scripts (pour le moment)** :

1. **links_collection.py**  
    Ce script parcourt le fichier "saved_links.p" qui contient les liens des parties annotées présentes dans la base de données de gameknot.com. En sortie, ce script donne une liste des liens présents dans le fichier.

2. **scraping_links.py**  
    Ce script effectue un scraping sur les liens obtenus avec "links_collection.py". Il produit en sortie un dossier "corpus_csv" contenant un fichier CSV par partie, nommés progressivement "df_match_1", "df_match_2", etc. Chaque fichier CSV contient les colonnes suivantes :
    - **N_move** (1, 2, 3,...) : indexation de chaque coup ;
    - **Standard_notation** (e4, f5,...) : notation standard qui décrit la case où la pièce a été déplacée ;
    - **UCI_notation** (e2e4, f7f5,...) : notation UCI qui décrit la case où la pièce a été déplacée + la case de départ ;
    - **FEN_notation** (rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1) : décrit l'état de l'échiquier après le coup ;
    - **Comment** (jusqu'à présent, tout va bien...) : commentaire relatif au coup.

3. **download_BART_model.py**  
    Ce script télécharge localement le modèle BART "facebook/bart-large" et son Tokenizer.

4. **pre_processing.py**  
    Ce script contient :
    - Une fonction qui renvoie le chemin du dossier contenant les csv des matchs (find_corpus_folder) ;
    - Une fonction qui renvoie, pour une notation FEN en entrée, une notation FEN encodée (encode_fen) ;
    - Une fonction qui renvoie, pour un commentaire en entré, un commentaire encodée (encode_comment).

5. **model_test_1.py**  
    Ce script utilise le modèle BART pour générer un commentaire à partir d'une notation FEN donnée en entrée. C'est une version de base. Pour le moment, la génération est de très mauvaise qualité et le modèle n'est pas entraîné sur les données contenues dans le dossier "corpus_csv".

6. **model_test_2.py**
    ...Work in progress...
    Tentafin de fine-tuning de BART avec X = FENs et y = commentaire. Script incomplet.