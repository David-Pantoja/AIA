import requests

# Properly format the CIK with leading zeros (total 10 digits)
cik = "0000789019"
url = f"https://data.sec.gov/submissions/CIK{cik}.json"

headers = {
    "User-Agent": "David Pantoja davidpantoja2727@gmail.com", # Replace with your information
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Successfully fetched data")
    print(data)
    # Process data here
else:
    print(f"Error: {response.status_code}")
    print(f"Response content: {response.text[:500]}")