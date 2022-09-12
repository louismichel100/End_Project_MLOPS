import requests

url = 'http://127.0.0.1:9696/predict'

ride = {"debut_signal": 0, "numero_personne": 2}

response = requests.post(url, json=ride).json()
print(response)
