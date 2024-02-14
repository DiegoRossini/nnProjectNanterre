# Téléchargement du modèle BART et de son tokeniseurmodel
from download_BART_model import download_model, download_tokenizer
model = download_model()
tokenizer = download_tokenizer()


# Importations depuis pre_processing.py
from pre_processing import *
import torch



'''
-------------- EXPERIENCE 3 : ENTRAINER LE MODELE A JOUER AUX ECHECS ------------------------
'''


# N_move,Standard_notation,UCI_notation,FEN_notation,Comment

def build_dataset():

    # Constitution de X et y vides
    X = []
    y = []

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    # corpus_path = corpus_path + "/*.csv"
    corpus_path = corpus_path + "/df_match_0.csv"

    # Parcours tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        for idx, fen in enumerate(df['FEN_notation']):
            X.append(fen)
            next_move = df['Standard_notation'][idx+1]
            print(next_move)
            
            #y.append(next_move)

    #return X, y

build_dataset()
