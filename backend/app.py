from fastapi import FastAPI


app = FastAPI()

@app.get("/")
async def home():
    # Todo: Implement the home endpoint
    return {""}

@app.get("/predict-players")
async def predict_players():
    return {}

@app.get("/get-games-today")
async def get_games_today():
    return {}

@app.get("/get-player-info")
async def get_player_info():
    return {}

@app.get("/get-player-stats")
async def get_player_stats():
    return {}



