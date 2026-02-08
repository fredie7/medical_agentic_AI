# import os
# import json
# from dotenv import load_dotenv
# import requests
# import pandas as pd

# load_dotenv()

# def get_teams_table():
#     try:
#         headers = {'X-Auth-Token': os.getenv("api_key")}
#         url = "https://api.football-data.org/v4/competitions/PL/teams"
#         response = requests.get(url=url, headers=headers)
#         data = response.json()

#         # Extract teams info
#         teams_list = []
#         for team in data['teams']:
#             teams_list.append({
#                 "team_id": team['id'],
#                 "name": team['name'],
#                 "short_name": team['shortName'],
#                 "tla": team['tla'],
#                 "crest_url": team['crest'],
#                 "founded": team.get('founded', None),
#                 "venue": team.get('venue', None),
#                 "address": team.get('address', None),
#                 "website": team.get('website', None),
#                 "coach": team.get('coach', {}).get('name', None),
#                 "coach_nationality": team.get('coach', {}).get('nationality', None)
#             })

#         # Convert to pandas DataFrame for nice table format
#         df = pd.DataFrame(teams_list)
#         print(df)
#         return df

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# # Call the function
# get_teams_table()

import os
import json
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()

def get_teams_with_players():
    try:
        headers = {'X-Auth-Token': os.getenv("api_key")}
        url = "https://api.football-data.org/v4/competitions/PL/teams"
        response = requests.get(url=url, headers=headers)
        data = response.json()

        teams_list = []

        for team in data['teams']:
            # Extract players from the squad
            players = team.get('squad', [])
            player_names = [player.get('name') for player in players]

            teams_list.append({
                "team_id": team['id'],
                "name": team['name'],
                "short_name": team['shortName'],
                "tla": team['tla'],
                "crest_url": team['crest'],
                "founded": team.get('founded', None),
                "venue": team.get('venue', None),
                "address": team.get('address', None),
                "website": team.get('website', None),
                "coach": team.get('coach', {}).get('name', None),
                "coach_nationality": team.get('coach', {}).get('nationality', None),
                "players": player_names  # List of player names
            })

        # Convert to pandas DataFrame
        df = pd.DataFrame(teams_list)
        print(df)
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Call the function
df_teams_players = get_teams_with_players()
