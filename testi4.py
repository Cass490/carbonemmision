from google.colab import files
import pandas as pd
import io

# Upload the file
uploaded = files.upload()

# Read the uploaded file into a pandas DataFrame
for filename in uploaded.keys():
    df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Display the DataFrame
print("File uploaded successfully!")
df.head()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def predict_and_plot(df, future_years):
    X = df[['Year']]
    y = df['CO2_Emissions']

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predicted_emissions = model.predict(future_years)

    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(future_years, predicted_emissions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predicted Data')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (Kilotons)')
    plt.title('CO2 Emissions Prediction')
    plt.legend()
    plt.show()

# Future years to predict
future_years = pd.DataFrame({'Year': [2023, 2024, 2025]})

# Predict and plot
predict_and_plot(df, future_years)
