# # import requests
# # import os
# # import csv
# # from dotenv import load_dotenv

# # load_dotenv()

# # def top_scorers():
# #     try:
# #         headers = {'X-Auth-Token': os.getenv("api_key")}
# #         url = "https://api.football-data.org/v4/competitions/PL/scorers"
# #         response = requests.get(url=url, headers=headers)
# #         data = response.json()
        
# #         # Extract top scorers info
# #         scorers = data.get('scorers', [])
        
# #         # CSV file
# #         with open('top_scorers.csv', mode='w', newline='') as file:
# #             writer = csv.writer(file)
# #             writer.writerow([
# #                 "Player ID", "Full Name", "First Name", "Last Name", "Date of Birth",
# #                 "Nationality", "Position / Section", "Played Matches", "Goals",
# #                 "Assists", "Penalties", "Season", "Competition Name", 
# #                 "Competition Code", "Player Last Updated", "Team Last Updated"
# #             ])
            
# #             for item in scorers:
# #                 player = item.get('player', {})
# #                 team = item.get('team', {})
                
# #                 writer.writerow([
# #                     player.get('id'),
# #                     player.get('name'),
# #                     player.get('firstName'),
# #                     player.get('lastName'),
# #                     player.get('dateOfBirth'),
# #                     player.get('nationality'),
# #                     player.get('section'),
# #                     item.get('playedMatches'),
# #                     item.get('goals'),
# #                     item.get('assists'),
# #                     item.get('penalties'),
# #                     data.get('filters', {}).get('season'),
# #                     data.get('competition', {}).get('name'),
# #                     data.get('competition', {}).get('code'),
# #                     player.get('lastUpdated'),
# #                     team.get('lastUpdated')
# #                 ])
        
# #         print("Top scorers exported to 'top_scorers.csv'")
# #         print(data)
# #         return data

# #     except Exception as e:
# #         print(f"An error occurred: {e}")
# #         return None

# # top_scorers()

# import requests
# import os
# import pandas as pd
# from dotenv import load_dotenv

# load_dotenv()

# def top_scorers():
#     try:
#         headers = {'X-Auth-Token': os.getenv("api_key")}
#         url = "https://api.football-data.org/v4/competitions/PL/scorers"
#         response = requests.get(url=url, headers=headers)
#         data = response.json()
        
#         scorers = data.get('scorers', [])
        
#         # Prepare list of rows
#         rows = []
#         for item in scorers:
#             player = item.get('player', {})
#             team = item.get('team', {})
            
#             rows.append({
#                 "Player ID": player.get('id'),
#                 "Full Name": player.get('name'),
#                 "First Name": player.get('firstName'),
#                 "Last Name": player.get('lastName'),
#                 "Date of Birth": player.get('dateOfBirth'),
#                 "Nationality": player.get('nationality'),
#                 "Position / Section": player.get('position') or player.get('section'),
#                 "Played Matches": item.get('playedMatches'),
#                 "Goals": item.get('goals'),
#                 "Assists": item.get('assists'),
#                 "Penalties": item.get('penalties'),
#                 "Season": data.get('filters', {}).get('season'),
#                 "Competition Name": data.get('competition', {}).get('name'),
#                 "Competition Code": data.get('competition', {}).get('code'),
#                 "Player Last Updated": player.get('lastUpdated'),
#                 "Team Last Updated": team.get('lastUpdated')
#             })
        
#         # Convert to Pandas DataFrame
#         df = pd.DataFrame(rows)
#         print(df)  # Show in table format
#         return df

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# # Run the function
# df_top_scorers = top_scorers()

import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def top_scorers():
    try:
        headers = {'X-Auth-Token': os.getenv("api_key")}
        url = "https://api.football-data.org/v4/competitions/PL/scorers"
        response = requests.get(url=url, headers=headers)
        data = response.json()
        
        scorers = data.get('scorers', [])
        rows = []
        
        for item in scorers:
            player = item.get('player', {})
            team = item.get('team', {})
            
            rows.append({
                # Player info
                "Player ID": player.get('id'),
                "Full Name": player.get('name'),
                "First Name": player.get('firstName'),
                "Last Name": player.get('lastName'),
                "Date of Birth": player.get('dateOfBirth'),
                "Nationality": player.get('nationality'),
                "Position / Section": player.get('position') or player.get('section'),
                "Shirt Number": player.get('shirtNumber'),
                "Player Last Updated": player.get('lastUpdated'),
                
                # Team info
                "Team ID": team.get('id'),
                "Team Name": team.get('name'),
                "Team Short Name": team.get('shortName'),
                "Team TLA": team.get('tla'),
                "Team Crest": team.get('crest'),
                "Team Venue": team.get('venue'),
                "Team Founded": team.get('founded'),
                "Team Website": team.get('website'),
                "Team Last Updated": team.get('lastUpdated'),
                
                # Stats
                "Played Matches": item.get('playedMatches'),
                "Goals": item.get('goals'),
                "Assists": item.get('assists'),
                "Penalties": item.get('penalties'),
                
                # Competition / Season info
                "Season": data.get('filters', {}).get('season'),
                "Competition Name": data.get('competition', {}).get('name'),
                "Competition Code": data.get('competition', {}).get('code')
            })
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(rows)
        print(df)  # Display the table
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Run the function
df_top_scorers = top_scorers()
