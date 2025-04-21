from tensorflow.keras import layers, Input, Model

def build_cnn_ffnn_binary(image_shape, metadata_shape):
    img_input = Input(shape=image_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(img_input)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    meta_input = Input(shape=(metadata_shape,))
    m = layers.Dense(64, activation='relu')(meta_input)
    m = layers.Dense(32, activation='relu')(m)

    combined = layers.concatenate([x, m])
    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(z)  # Binary classification

    model = Model(inputs=[img_input, meta_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

