# # Assurez-vous que la GPU est disponible
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Téléchargement du modèle BART et de son tokeniseur
from download_BART_model import download_tokenizer
tokenizer = download_tokenizer()


# Importations
import os
import glob
import pandas as pd


# Fonction pour trouver le dossier du corpus
def find_corpus_folder(directory='corpus_csv'):

    # Parcours le répertoire spécifié (par défaut : répertoire 'corpus_csv')
    for root, dirs, files in os.walk(os.path.abspath(os.sep)):

        # Vérifie si le dossier cible est présent dans les répertoires actuels
        if directory in dirs:

            # Renvoie le chemin complet du dossier cible s'il est trouvé
            return os.path.join(root, directory)
        
    # Si le dossier cible n'est pas trouvé, renvoie None
    return None



# Function pour selectionner un corpus d'entrainement plus petit (que des matchs ayant au moins 50 coups et pas moins de 80)
def select_reduced_corpus(corpus_path, max_files=3000):

    # Initialisation d'une liste vide et d'un compteur
    selected_csv_files = []

    # Itération à travers chaque csv du corpus
    for csv_match_path in glob.glob(corpus_path):

        # Ouverture du csv et comptage des lignes
        with open(csv_match_path, 'r', encoding='utf-8') as file:
            line_count = sum(1 for line in file)

        # Check si le csv contient plus de 50 lignes et moins de 80
        if 50 <= line_count <= 80:
            selected_csv_files.append(csv_match_path)

            # Check si la taille du corpus d'entrainement est atteinte, si oui l'output sera la liste des parcours des csv
            if len(selected_csv_files) >= max_files:
                return selected_csv_files

    return selected_csv_files


'''
Les deux fonctions qui suivent servent à tokenizer et encoder un mouvement en notation FEN
'''


# Fonction pour obtenir le vocabulaire des mouvements en notation FEN
def get_FEN_vocab():

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = os.path.join(corpus_path, "*.csv")

    # Initialise le vocabulaire FEN comme un ensemble pour éviter les doublons
    FEN_vocab = set()

    # Parcourt tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)  

        # Parcourt toutes les notations FEN dans la colonne 'FEN_notation'
        for FEN in df['FEN_notation']:

            # Parcourt chaque caractère dans la notation FEN
            for car in FEN:

                # Ajoute le caractère au vocabulaire FEN
                FEN_vocab.add(car)      

    # Convertit FEN_vocab en une liste
    FEN_vocab = list(FEN_vocab)

    # Ajoute "<end>" au début de la liste
    FEN_vocab.insert(0, "<end_fen>")

    # Insère "<start>" au début de la liste
    FEN_vocab.insert(0, "<start_fen>")

    # Renvoie FEN_vocab
    return FEN_vocab



# Fonction pour générer une notation FEN encodée
def encode_fen(input_fen):

    # On tokenize la notation au niveau du caractère
    tokenized_fen = [car for car in input_fen]

    # On encode la notation FEN avec le tokenizer
    encoded_fen = tokenizer.encode_plus(tokenized_fen, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

    # Formattage pour s'adapter à l'entrée du modèle dans le batch
    encoded_fen = {key: value.squeeze(0) for key, value in encoded_fen.items()}

    # La sortie est une séquence d'entiers en tensor
    return encoded_fen



'''
Les quatre fonctions qui suivent servent à tokenizer et encoder un commentaire
'''


# Fonction pour obtenir le vocabulaire de tous les mouvements en notation standard effectués dans les matchs du corpus
def get_st_notation_vocab():

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = os.path.join(corpus_path, "*.csv")

    # Crée un vocabulaire des mouvements standard vide
    all_st_notation_vocab = set()

    # Parcourt tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path) 

        # Parcourt toutes les notations standard dans la colonne 'Standard_notation'
        for st_notation in df['Standard_notation']:
            all_st_notation_vocab.add(st_notation)

    # Retourne la liste de tous les mouvements du corpus en notation standard
    return list(all_st_notation_vocab)
        


# Fonction pour obtenir le vocabulaire spécifique des mouvements cités dans les commentaires et aider à la tokenisation de BART
def get_comments_st_notation_vocab(all_st_notation_vocab):

    # Initialise un vocabulaire vide pour les mouvements standard présents dans les commentaires
    '''
    Si on utilise all_standard_notation_vocab en tant qu'entrée pour la tokenisation du 'fine-tuned BART Tokenizer',
    on risque d'alourdir l'apprentissage car tous les mouvements standard ne sont pas présents dans les commentaires!
    Il est donc préférable d'utiliser le vocabulaire des mouvements standard présents dans les commentaires pour la tokenisation du 'fine-tuned BART Tokenizer'
    '''
    comments_st_notation_vocab = set()

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = os.path.join(corpus_path, "*.csv")

    # Parcourt tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        # Tente de boucler sur chaque commentaire
        try:
            # Boucle sur chaque commentaire du match
            for comment in df['Comment']:

                # Boucle sur chaque mouvement du corpus en notation standard
                for move in all_st_notation_vocab:

                    # Si un mouvement en notation standard est cité dans le commentaire
                    if move in comment:

                        # Alors on l'ajoute à notre liste
                        comments_st_notation_vocab.add(move)

        # Si le commentaire n'est pas une chaîne de caractères, on continue
        except:
            continue
    
    # En sortie, le vocabulaire de tous les mouvements standard présents dans les commentaires
    return list(comments_st_notation_vocab)



# Fonction qui tokenise avec BART Tokenizer un commentaire en entrée
def tokenize_comment(comment):

    # Tokenisation du commentaire avec le 'BART Tokenizer'
    tokenized_comment = tokenizer.tokenize(comment)

    # Retourne le commentaire tokenisé avec les caractères spéciaux propres au BART Tokenizer
    return tokenized_comment



# Fonction qui encode un commentaire en entrée avec BART Tokenizer
def encode_comment(tokenized_comment):

    # Encode le commentaire avec le 'BART Tokenizer'
    encoded_comment = tokenizer.encode_plus(tokenized_comment, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

    # Formattage pour s'adapter à l'entrée du modèle dans le batch
    encoded_comment = {key: value.squeeze(0) for key, value in encoded_comment.items()}

    # Retourne le commentaire encodé avec le 'BART Tokenizer'
    return encoded_comment



'''
Les deux fonctions qui suivent servent à tokenizer et encoder une UCI
'''

def get_uci_vocab():

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = os.path.join(corpus_path, "*.csv")

    # Initialise le vocabulaire UCI comme un ensemble pour éviter les doublons
    uci_vocab = set()

    # Parcourt tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)  

        # Parcourt toutes les notations UCI dans la colonne 'UCI_notation'
        for uci in df['UCI_notation']:

            # Parcourt chaque mouvement dans la notation UCI (donc deux caractères par deux caractères)
            case_depart = uci[:2]
            case_fin = uci[2:4] 

            # Ajoute le mouvement au vocabulaire UCI
            uci_vocab.add(case_depart)      
            uci_vocab.add(case_fin)      

    # Convertit uci_vocab en une liste
    uci_vocab = list(uci_vocab)

    # Ajoute "<end>" au début de la liste
    uci_vocab.insert(0, "<end_uci>")

    # Insère "<start>" au début de la liste
    uci_vocab.insert(0, "<start_uci>")

    return uci_vocab


def encode_uci(input_uci, uci_vocab):

    # On tokenize la notation au niveau du caractère
    tokenized_uci = [input_uci[:2], input_uci[2:4]]

    # On encode la notation UCI avec le tokenizer
    encoded_uci = tokenizer.encode_plus(tokenized_uci, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

    # Formattage pour s'adapter à l'entrée du modèle dans le batch
    encoded_uci = {key: value.squeeze(0) for key, value in encoded_uci.items()}

    # La sortie est une séquence d'entiers en tensor
    return encoded_uci