import os
import requests
from dotenv import load_dotenv
import ieugwaspy as igp
import json


def get_opengwas_jwt():
    if not os.path.exists(".ieugwaspy.json"):
        print("JWT token not found in .ieugwaspy.json. Running ieugwaspy.get_jwt() to set up authentication.")
        igp.get_jwt()

    with open(".ieugwaspy.json", "r") as f:
        token_data = json.load(f)

    token = token_data.get("jwt")
    if token is None:
        raise ValueError("JWT token not found in .ieugwaspy.json")

    return token


def get_user_data():
    url = "https://api.opengwas.io/api/user"

    token = get_opengwas_jwt()
    headers = {"Authorization": f"Bearer {token}"}

    print(f"Attempting GET request to {url}...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        raise ValueError("Unauthorized: Your token is invalid or expired. Set new token in .env file.")
    elif response.status_code == 404:
        raise ValueError("Endpoint not found: Ensure you are using the correct API URL.")
    else:
        raise ValueError(f"Unexpected error: {response.status_code} {response.text}")


if __name__ == "__main__":
    try:
        token = get_opengwas_jwt()
        user_data = get_user_data()
        print("User Data:", user_data)
    except ValueError as e:
        print(e)
