from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('model', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Mappings for categorical inputs
brand_map = {
    'maruti': 1, 'skoda': 2, 'honda': 3, 'hyundai': 4, 'toyota': 5,
    'ford': 6, 'renault': 7, 'mahindra': 8, 'tata': 9, 'chevrolet': 10,
    'datsun': 11, 'jeep': 12, 'benz': 13, 'mitsubishi': 14, 'audi': 15,
    'volkswagen': 16, 'bmw': 17, 'nissan': 18, 'lexus': 19, 'jaguar': 20,
    'land rover': 21, 'mg': 22, 'volvo': 23, 'daewoo': 24, 'kia': 25,
    'fiat': 26, 'force': 27, 'ambassador': 28, 'ashok': 29, 'isuzu': 30,
    'opel': 31
}

fuel_map = {
    'petrol': 1, 'diesel': 2, 'cng': 3, 'lpg': 4
}

seller_map = {
    'individual': 1, 'dealer': 2, 'trustmark d': 3
}

transmission_map = {
    'manual': 1, 'automatic': 2
}

owner_map = {
    'first owner': 1,
    'second owner': 2,
    'third owner': 3,
    'fourth & above owner': 4,
    'test drive car': 5
}

# Exchange rates (INR base)
exchange_rates = {
    'inr': 1,
    'lkr': 3.41,    # Example: 1 INR = 3.41 LKR
    'usd': 0.0112   # Example: 1 INR = 0.0112 USD
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    currency = "INR"  # Default

    if request.method == 'POST':
        try:
            # Collect form inputs
            brand = request.form['brand'].strip().lower()
            year = int(request.form['year'])
            kms = int(request.form['kms'])
            fuel = request.form['fuel'].strip().lower()
            seller = request.form['seller'].strip().lower()
            transmission = request.form['transmission'].strip().lower()
            owner = request.form['owner'].strip().lower()
            mileage = float(request.form['mileage'])
            engine = float(request.form['engine'])
            power = float(request.form['power'])
            seats = int(request.form['seats'])
            currency = request.form['currency'].strip().lower()

            # Convert categorical to numerical values safely
            brand_code = brand_map.get(brand)
            fuel_code = fuel_map.get(fuel)
            seller_code = seller_map.get(seller)
            transmission_code = transmission_map.get(transmission)
            owner_code = owner_map.get(owner)

            # Validate mappings
            if None in [brand_code, fuel_code, seller_code, transmission_code, owner_code]:
                error = "Invalid input detected in one of the dropdowns."
                return render_template('index.html', prediction=None, error=error, currency=currency.upper())

            # Build input feature array
            features = np.array([[brand_code, year, kms, fuel_code,
                                  seller_code, transmission_code,
                                  owner_code, mileage, engine, power, seats]])

            # Predict price (in INR)
            predicted_price_in_inr = model.predict(features)[0]

            # Ensure no negative prices
            predicted_price_in_inr = max(0, predicted_price_in_inr)

            # Convert to selected currency
            rate = exchange_rates.get(currency, 1)
            prediction = round(predicted_price_in_inr * rate, 2)

        except Exception as e:
            error = f"Something went wrong: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error, currency=currency.upper())

if __name__ == '__main__':
    app.run(debug=True)
