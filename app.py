from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Load trained model & scaler
model = pickle.load(open('rainfall_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        input_features = [float(request.form[col]) for col in ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                         'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]
        print("Received Inputs:", input_features)  # Debugging Line

        input_array = np.array(input_features).reshape(1, -1)
        print("Input Array Before Scaling:", input_array)


        # Scale input
        input_array = scaler.transform(input_array)
        print("Scaled Input Array:", input_array)

        # Predict rainfall probability
        prediction = model.predict_proba(input_array)[0][1]

        return render_template('index.html', prediction=f'Rainfall Probability: {prediction:.2f}')
    except Exception as e:
        print("Error Occurred:", str(e))  # Print actual error in console
        return render_template('index.html', prediction=f"Error: {str(e)}")  # Show real error in UI

if __name__ == '__main__':
    app.run(debug=True)