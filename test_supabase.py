import requests
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

headers = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}

url = f"{SUPABASE_URL}/rest/v1/profiles"

response = requests.get(url, headers=headers, verify=False)

print("Status Code:", response.status_code)
print("Response Body:", response.json())