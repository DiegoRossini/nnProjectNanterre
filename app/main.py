from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import chess
import chess.svg
from model_test_1 import comment_generation_model_test_1

app = FastAPI()

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates instance
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render the HTML form using Jinja template
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit_fen")
async def submit_fen(request: Request):
    # Get the form data submitted by the user
    form_data = await request.form()
    fen = form_data["fen"]

    # Generate the comment
    comment = comment_generation_model_test_1(fen)
    
    print("comment : ", comment)

    # Return the comment as a response
    return {"comment": comment}

    # # Return a response
    # return {"message": f"Chess fen '{fen}' submitted successfully."}



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