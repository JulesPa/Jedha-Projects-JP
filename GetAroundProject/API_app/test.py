import requests

url = "https://getaround-api-app-82984add3f24.herokuapp.com/predict"
payload = {
    "input": [["Citroën", 140411, 100, "diesel", "black", "convertible", True, True, False, False, True, True, True]]
}
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)  # Print the status code
print("Response Text:", response.text)       # Print the full response text

# If the status code is 200, parse the JSON
if response.status_code == 200:
    print(response.json())
else:
    print("Error: API did not return a successful response")