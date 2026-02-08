import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

# api_key = os.getenv("api_key")
# print(api_key)
# api_host = os.getenv("api_host")
# print(api_host)
# league_id = 39  # Example league ID (Premier League)
# season = 2025   # Example season year
# url = "https://api-football-v1.p.rapidapi.com/v3/standings"
# headers = {
#     'x-rapidapi-key': api_key,
#     'x-rapidapi-host': api_host
# }

# querystring = {"league": str(league_id), "season": str(season)}
# # print(querystring)
# response = requests.get(url=url, headers=headers, params=querystring)
# data = response.json()
# print(data)

# def get_league_standing():
#     try:
#         headers = {'X-Auth-Token': os.getenv("api_key")}
#         url = "https://api.football-data.org/v4/competitions/PL/standings"
#         response = requests.get(url=url, headers=headers)
#         data = response.json()
#         print(data)
#         return data
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
# get_league_standing()

def get_league_standing():
    try:
        headers = {'X-Auth-Token': os.getenv("api_key")}
        url = "https://api.football-data.org/v4/competitions/PL/standings"
        response = requests.get(url=url, headers=headers)
        data = response.json()

        standings = data["standings"][0]["table"]

        print("\nPremier League Standings\n")
        print(f"{'Pos':<4} {'Team':<25} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'Pts':<4}")
        print("-" * 50)

        for entry in standings:
            pos = entry["position"]
            team = entry["team"]["name"]
            played = entry["playedGames"]
            won = entry["won"]
            draw = entry["draw"]
            lost = entry["lost"]
            points = entry["points"]

            print(f"{pos:<4} {team:<25} {played:<3} {won:<3} {draw:<3} {lost:<3} {points:<4}")

        return standings

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
get_league_standing()