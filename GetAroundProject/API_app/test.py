import requests

url = "https://getaround-api-app-82984add3f24.herokuapp.com/predict"
payload = {
    "input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]
}
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)  
print("Response Text:", response.text)       


if response.status_code == 200:
    print(response.json())
else:
    print("Error: API did not return a successful response")
