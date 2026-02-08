import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("api_key")
print(api_key)
api_host = os.getenv("api_host")
print(api_host)
league_id = 39  # Example league ID (Premier League)
season = 2025   # Example season year
url = "https://api.football-data.org/v4/competitions/PL/teams"
headers = {
    'x-rapidapi-key': api_key,
    'x-rapidapi-host': api_host
}

querystring = {"league": str(league_id), "season": str(season)}
# print(querystring)
response = requests.get(url=url, headers=headers, params=querystring)
data = response.json()
print(data)

def get_team_details():
    try:
        headers = {'X-Auth-Token': os.getenv("api_key")}
        url = "https://api.football-data.org/v4/competitions/PL/teams"
        response = requests.get(url=url, headers=headers)
        data = response.json()
        print(data)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
get_team_details()