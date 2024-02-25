# Le fichier "saved_links.p" a été recuperé depuis "https://github.com/harsh19/ChessCommentaryGeneration". Voir le README pour plus d'informations.

# Importations
import pickle
import os
import re

# Une fonction parcourant le système à la recherche de "saved_links.p".
def find_saved_links_file():
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file == "saved_links.p":
                return os.path.join(root, file)

# Une fonction qui utilise les expressions régulières pour extraire les liens https
def extract_links_from_pickle():

    # Affectation du lien du fichier pickle à une variable
    file_path = find_saved_links_file()

    # Liste pour stocker les liens
    links = []

    try:
        # Ouverture du fichier pickle en mode lecture binaire
        with open(file_path, 'rb') as f:

            # Chargement des données du fichier pickle
            data = pickle.load(f)
            
            # Regex pour capturer les liens
            regex_pattern = r'https:\/\/gameknot\.com\/annotation\.pl\/[^\s]+'

            # Trouve tous les liens dans le texte
            for item in data:
                links.extend(re.findall(regex_pattern, item))

    except FileNotFoundError:
        print(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors de l'extraction des liens: {e}")

    return links