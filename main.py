import requests
import dotenv
import os

dotenv.load_dotenv()

MESSAGE = 'Hello from HackMIT!'
API_KEY = os.getenv('POKE_API_KEY')

response = requests.post(
    'https://poke.com/api/v1/inbound-sms/webhook',
    headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    },
    json={'message': MESSAGE}
)

print(response.json())