from tensorflow.keras.models import load_model
import numpy as np
model = load_model('/content/modelo_mejorado.h5')

players_data = np.array([
    [2, 28, 30, 1, 0, 0, 9, 1, 1.7],
    [4, 33, 32, 27, 12, 7, 0, 0, 23.5],
    [4, 32, 34, 35, 3, 5, 2, 0, 32],
    [2, 31, 18, 0, 0, 0, 4, 1, 0.5]
])

scaler = StandardScaler()
players_data_scaled = scaler.fit_transform(players_data)

predictions = model.predict(players_data_scaled)

for i, prediction in enumerate(predictions):
    print(f"Jugador {i+1}: {'Contratar' if prediction >= 0.2 else 'No contratar'} (Probabilidad: {prediction[0]:.2f})")
