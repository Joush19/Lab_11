from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
data = pd.read_excel('/content/Jugadores.xlsx')

features = ['Posicion Num', 'Edad', 'Partidos Jugados', 'Goles', 'Asistencias',
            'Penales Acertados', 'Tarjetas Amarillas', 'Tarjetas Rojas', 'Goles Esperados']
X = data[features]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.005),
    Dense(32, activation='relu'),
    Dropout(0.005),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.00005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

model.save('/content/modelo_mejorado.h5')
