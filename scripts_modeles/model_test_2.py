# Téléchargement du modèle BART et de son tokeniseur
# from download_BART_model import download_model
# model, tokenizer = download_model()



# Téléchargements des fonctions de prétraitement nécessaires
from pre_processing import find_corpus_folder


# Importation des librairies nécessaires
# from transformers import BartConfig
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import numpy as np
from sklearn.model_selection import train_test_split



# Fonction d'extraction de X et de y
def get_X_y():


    X = []
    y = []

    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')

    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = corpus_path + "/*.csv"

    # Parcours tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)

        for fen, comment in map(df['FEN_notation'], df['Comment']):
            X.append(fen)
            y.append(comment)

    return X, y

# Définition de vos données d'entraînement et de test
train_texts = [...]  # Liste des textes d'entraînement
train_labels = [...]  # Liste des étiquettes d'entraînement

test_texts = [...]  # Liste des textes de test
test_labels = [...]  # Liste des étiquettes de test

# Tokenisation des données d'entraînement et de test
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Création des DataLoader pour les données d'entraînement et de test
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_labels))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_encodings['attention_mask']),
                             torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Définition de l'optimiseur
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Fonction de perte
criterion = torch.nn.CrossEntropyLoss()

# Entraînement du modèle
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Évaluation du modèle
model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        all_predictions.extend(outputs)
        all_labels.extend(labels)

# Calcul des métriques d'évaluation
accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
print(f'Précision : {accuracy}')