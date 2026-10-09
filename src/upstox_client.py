import os
import requests
from dotenv import load_dotenv

load_dotenv()

UPSTOX_TOKEN = os.getenv("UPSTOX_ANALYTICS_TOKEN")

BASE_URL = "https://api.upstox.com"

HEADERS = {
    "Authorization": f"Bearer {UPSTOX_TOKEN}",
    "Accept": "application/json",
}


def get(endpoint, params=None):
    """
    Send an authenticated GET request to the Upstox API.
    """

    if not UPSTOX_TOKEN:
        raise RuntimeError(
            "UPSTOX_ANALYTICS_TOKEN not found in .env"
        )

    url = BASE_URL + endpoint

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    return response.json()