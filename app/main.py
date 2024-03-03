# Importations nécessaires
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import chess
import chess.svg
# from model_test_1 import comment_generation_model_test_1
from mbart_comment import comment_generation

# Création de l'application FastAPI
app = FastAPI()

# Monter le répertoire statique pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialisation de l'instance Jinja2Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Rendre le HTML en utilisant le modèle Jinja
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def home(request: Request):
    # Rendre le HTML en utilisant le modèle Jinja
    return templates.TemplateResponse("about.html", {"request": request})

@app.post("/submit_fen")
async def submit_fen(request: Request):
    # Obtenir les données du formulaire soumises par l'utilisateur
    form_data = await request.form()
    fen = form_data["fen"]

    # Générer le commentaire
    comment = comment_generation(fen)
    
    print("commentaire : ", comment)

    # Renvoyer le commentaire en tant que réponse
    return {"commentaire": comment}

# Configuration de Jinja
env = Environment(
    loader=FileSystemLoader('./templates'),
    autoescape=select_autoescape(['html', 'xml'])
)

@app.get("/board", response_class=HTMLResponse)
async def render_chess_board():
    board = chess.Board()
    svg_board = chess.svg.board(board=board)
    
    # Rendu du tableau de jeu avec Jinja
    template = env.get_template('game.html')
    rendered_board = template.render(board=svg_board)
    
    return rendered_board