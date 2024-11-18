To use the API type this command in your terminal :

    $ curl -i -H "Content-Type: application/json" -X POST -d '{"input": "input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]}' https://getaround-api-app-82984add3f24.herokuapp.com/predict

Or use Python:

    import requests

    response = requests.post("https://getaround-api-app-82984add3f24.herokuapp.com/predict", json={
        "input": "input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]
    })
    print(response.json())

Input should have the form (see exemple above):

     "input": [["model", mileage	, engine_power, "fuel", "paint_color", "car_type", private_parking_available, has_gps, has_air_conditioning, automatic_car, has_getaround_connect, has_speed_regulator, winter_tires]]
    
