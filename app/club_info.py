import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")
api_host = os.getenv("api_host")

def get_team_details(league="PL", season=2025):
    """
    Fetch team details from Football-Data API and return as a pandas DataFrame
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

        # Extract relevant team info
        teams_list = []
        for team in data.get("teams", []):
            teams_list.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "shortName": team.get("shortName"),
                "TLA": team.get("tla"),
                "crest": team.get("crest"),
                "address": team.get("address"),
                "website": team.get("website"),
                "founded": team.get("founded"),
                "clubColors": team.get("clubColors"),
                "venue": team.get("venue"),
                "runningCompetitions": [comp.get("name") for comp in team.get("runningCompetitions", [])]
            })

        # Convert to pandas DataFrame
        df = pd.DataFrame(teams_list)
        print(df)
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Call the function
df_teams = get_team_details()
print(df_teams)
