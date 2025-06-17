import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# Sample categories based on your original training data
geo_data = pd.DataFrame({
    'Geography': ['France', 'Germany', 'Spain']
})

# Create and fit OneHotEncoder
onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
onehot_encoder_geo.fit(geo_data[['Geography']])

# Save the encoder
with open('onehot_encoder_geo.pk1', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)

print("✅ OneHotEncoder for Geography saved successfully.")
