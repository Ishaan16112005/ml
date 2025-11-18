def predict_rain_simple():
    print("\nEnter today's weather values:")

    # Take inputs
    cloud_cover = float(input("Cloud Cover: "))
    sunshine = float(input("Sunshine: "))
    global_radiation = float(input("Global Radiation: "))
    max_temp = float(input("Max Temperature: "))
    mean_temp = float(input("Mean Temperature: "))
    min_temp = float(input("Min Temperature: "))
    precipitation = float(input("Precipitation: "))
    pressure = float(input("Pressure: "))
    snow_depth = float(input("Snow Depth: "))

    # Convert to array
    input_data = np.array([[
        cloud_cover,
        sunshine,
        global_radiation,
        max_temp,
        mean_temp,
        min_temp,
        precipitation,
        pressure,
        snow_depth
    ]])

    # Predict
    result = best_rf.predict(input_data)[0]

    # Output
    if result == 1:
        print("\n🌧️ Rain Tomorrow: YES")
    else:
        print("\n☀️ Rain Tomorrow: NO")
predict_rain_simple()
