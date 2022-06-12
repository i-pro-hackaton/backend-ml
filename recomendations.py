from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense, LSTM
from tensorflow.keras.models import Model

clicked_products = Input(name = 'clicked products', shape = [1])
other_criteries = Input(name = 'otner criteries', shape = [10])
criteries = Dense(10, activation='softmax')(other_criteries)
clicked_embedding = Embedding(name = 'clicked_embedding',
                           input_dim = 1000,
                           output_dim = 10)(clicked_products)
x = LSTM(units = 50, return_sequences = True)(clicked_embedding)
x = LSTM(units = 20, return_sequences = True)(x)
x = LSTM(units = 10, return_sequences = True)(x)
x = Reshape((10,))(x)
merged = Dot(name = 'dot_product', normalize = True, axes = 1)([x, criteries])
merged = Reshape(target_shape = [1])(merged)
out = Dense(1, activation = 'sigmoid')(merged)
model = Model(inputs = [clicked_products, other_criteries], outputs = out)
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
