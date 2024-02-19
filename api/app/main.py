from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import chess
import chess.svg

app = FastAPI()

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")




# Initialize Jinja2Templates instance
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render the HTML form using Jinja template
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit_move")
async def submit_move(request: Request):
    # Get the form data submitted by the user
    form_data = await request.form()
    move = form_data["move"]

    # Process the chess move (you can implement your logic here)

    # Return a response
    return {"message": f"Chess move '{move}' submitted successfully."}



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
    template = env.get_template('index.html')
    rendered_board = template.render(board=svg_board)
    
    return rendered_board
