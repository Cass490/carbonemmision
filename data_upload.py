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
