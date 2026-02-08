import requests

def search_web(topic):
    url = requests.get(f"https://api.duckduckgo.com/?q={topic}&format=json")
    print("Web Search Result:", url.json())
print(search_web("Artificial Intelligence"))