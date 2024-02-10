# Importations nécessaires
from bs4 import BeautifulSoup
import requests
import re
import chess
import pandas as pd
import os
from links_collection import extract_links_from_pickle


# Fonction qui prend en entrée une liste de liens et qui retourne un fichier csv contenant les mouvements en URI, FEN et commentaires des matchs.
def get_csv_from_links():

    # Recupération des liens
    links = extract_links_from_pickle()

    # Les csv seront sauvegardés dans le dossier "corpus_csv" qui sera contenu dans le même dossier d'où ce script est lancé
    os.mkdir(os.path.join(os.getcwd(),("corpus_csv")))
    corpus_directory = os.getcwd()+"/corpus_csv/"

    # Boucle sur chaque lien
    for n_match,url in enumerate(links):
        
        # Requete HTTP
        response = requests.get(url)

        # Vérification reponse
        if response.status_code == 200:

            # Parsing de la page
            soup = BeautifulSoup(response.text, 'html.parser')
            td_tags_moves = soup.find_all('td', {'rowspan': '2', 'style': 'vertical-align: top; width: 20%;'})

            # Extraction des mouvements
            moves_list = []
            for td_tags_move in td_tags_moves:
                move_content = td_tags_move.get_text(strip=True)
                moves_list.append(move_content)

            # Estraction des commentaires
            comments_list = []
            td_tags_comments = soup.find_all('td', {'style': 'vertical-align: top;'})
            for td_tags_comment in td_tags_comments:
                comment_content = td_tags_comment.get_text(strip=True)
                comments_list.append(comment_content)

            # Estraction index des commentaires
            script_tags = soup.find_all('script', {'type': 'text/javascript'})
            for content in script_tags:
                if 'anno_moves' in content.get_text():
                    matches = re.search(r'new Array\((.*?)\)', content.get_text(), re.DOTALL)
                    if matches:
                        comment_idxs = matches.group(1)
                        comment_idxs = comment_idxs.replace("'", "")
                        comment_idxs = comment_idxs.split(',')
                        comment_idxs = [int(idx) for idx in comment_idxs]
                    else:
                        print("Pas d'idx trouvés")

        # Si la reponse n'est pas 200, continuer à l'itération suivante
        else:
            print(f"Erreur de requete HTTP. Reponse au url {url} : {response}")
            continue
            
        # Création d'une liste des mouvements format couple (ex : 'e4 c5')
        individual_moves_couple = ' '.join(moves_list)
        individual_moves_couple = re.split(r'\d+\.\xa0', individual_moves_couple)[1:]
        individual_moves_couple = [move.strip() for move in individual_moves_couple]
        individual_moves_couple = [move.replace('\xa0', ' ').replace('.', '').strip() for move in individual_moves_couple]
        individual_moves_couple = [' '.join([word for word in move.split() if not word.isdigit()]) for move in individual_moves_couple]

        # Création d'une liste des mouvements format singulier (ex : 'e4', 'c5')
        individual_moves_single = []
        for individual_move_couple in individual_moves_couple:
            individual_moves_single.extend(individual_move_couple.split())

        # Conversion notation standard en UCI si des mouvements legeaux sont écrits dans le match en analyse (autrement on continue à l'intération suivante)
        try:
            uci_individual_moves_single = []
            board = chess.Board()
            for move in individual_moves_single:
                san_text = move
                uci_text = board.push_san(san_text).uci()
                uci_individual_moves_single.append(uci_text)
        except ValueError:
            print(f"Mouvement non standard dans le match {n_match}")
            continue

        # Conversion de UCI en FEN
        board = chess.Board()
        fen_moves_after_each_move = []
        for move_uci in uci_individual_moves_single:
            board.push(chess.Move.from_uci(move_uci))
            fen_moves_after_each_move.append(board.fen())

        # Création liste des commentaires pour chaque mouvement
        new_comment_list = []

        # Si le match n'existe plus alors une exception est levée, on continue à l'itération suivante
        try:
            actual_comment = comments_list[0]
            comment_idx = 0
            for idx in range(len(individual_moves_single)):
                if idx in comment_idxs:
                    comment_idx += 1
                    actual_comment = comments_list[comment_idx]
                    new_comment_list.append(actual_comment)
                else:
                    new_comment_list.append(actual_comment)
        except IndexError:
            print(f"Match {n_match} inexistant")
            continue

        # Création DataFrame
        df = pd.DataFrame({
            'N_move' : [idx + 1 for idx in range(len(uci_individual_moves_single))],
            'Standard_notation' : individual_moves_single,
            'UCI_notation' : uci_individual_moves_single,
            'FEN_notation' : fen_moves_after_each_move,
            'Comment' : new_comment_list
        })

        # Sauvegarde dans un csv
        df_match = f'df_match_{n_match}.csv'
        df_path = os.path.join(corpus_directory, df_match)
        df.to_csv(df_path, index=False)

        print(n_match)

get_csv_from_links()