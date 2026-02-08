import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_pl_standings():
    try:
        headers = {'X-Auth-Token': os.getenv("api_key")}
        url = "https://api.football-data.org/v4/competitions/PL/standings"
        response = requests.get(url=url, headers=headers)
        data = response.json()
        
        standings = data.get('standings', [])
        rows = []
        
        for standing in standings:
            # We are interested in TOTAL table
            if standing.get('type') == 'TOTAL':
                table = standing.get('table', [])
                for team_info in table:
                    team = team_info.get('team', {})
                    rows.append({
                        "Position": team_info.get('position'),
                        "Team ID": team.get('id'),
                        "Team Name": team.get('name'),
                        "Team Short Name": team.get('shortName'),
                        "Team TLA": team.get('tla'),
                        "Team Crest": team.get('crest'),
                        "Played Games": team_info.get('playedGames'),
                        "Won": team_info.get('won'),
                        "Draw": team_info.get('draw'),
                        "Lost": team_info.get('lost'),
                        "Points": team_info.get('points'),
                        "Goals For": team_info.get('goalsFor'),
                        "Goals Against": team_info.get('goalsAgainst'),
                        "Goal Difference": team_info.get('goalDifference')
                    })
        
        df = pd.DataFrame(rows)
        print(df)
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Get the DataFrame
df_standings = get_pl_standings()
