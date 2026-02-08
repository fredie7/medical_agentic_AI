import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")

def get_player_info(league="PL", season=2025):
    """
    Fetch player info from Football-Data API for all teams in a league and season.
    Returns a pandas DataFrame.
    """
    url = f"https://api.football-data.org/v4/competitions/{league}/teams"
    headers = {
        'X-Auth-Token': api_key
    }
    params = {"season": season}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        players_list = []
        for team in data.get("teams", []):
            squad = team.get("squad", [])
            for player in squad:
                players_list.append({
                    "playerID": player.get("id"),
                    "playerName": player.get("name"),
                    "position": player.get("position"),
                    "dateOfBirth": player.get("dateOfBirth"),
                    "nationality": player.get("nationality"),
                    "teamName": team.get("name")  # optional: which team the player belongs to
                })

        # Convert to pandas DataFrame
        df_players = pd.DataFrame(players_list)
        print(df_players)
        return df_players

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Call the function
df_players = get_player_info()
