from keras.models import load_model
import numpy as np
model = load_model('/content/modelo_mnist_cnn.h5')

x_test_selected = np.array([x_test[1], x_test[12], x_test[200]])

x_test_selected = np.expand_dims(x_test_selected, axis=-1)
x_test_selected = x_test_selected / 255.0

predictions = model.predict(x_test_selected)

for i, prediction in enumerate(predictions):
    predicted_label = np.argmax(prediction)
    expected_label = [2, 9, 3]
    print(f"Imagen {i+1}: Predicción: {predicted_label} (Esperado: {expected_label[i]})")
