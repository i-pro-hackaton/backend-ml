from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense
from tensorflow.keras.models import Model

bought_products = Input(name = 'bought products', shape = [1])
skipped_products = Input(name = 'skipped products', shape = [1])

bought_embedding = Embedding(name = 'bought_embedding',
                           input_dim = 1000,
                           output_dim = 10)(bought_products)
skipped_embedding = Embedding(name = 'skipped_embedding',
                           input_dim = 1000,
                           output_dim = 10)(skipped_products)
merged = Dot(name = 'dot_product', normalize = True, axes = 2)([bought_embedding, skipped_embedding])
merged = Reshape(target_shape = [1])(merged)
out = Dense(1, activation = 'sigmoid')(merged)
model = Model(inputs = [bought_products, skipped_products], outputs = out)
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
