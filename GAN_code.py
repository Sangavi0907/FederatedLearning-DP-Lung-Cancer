import os, numpy as np, cv2, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

def build_generator():
    model = keras.Sequential([
        layers.Dense(16*16*128, activation="relu", input_dim=100),
        layers.Reshape((16, 16, 128)),
        layers.UpSampling2D(),
        layers.Conv2D(128, kernel_size=3, padding="same"),
        layers.BatchNormalization(), layers.ReLU(),
        layers.UpSampling2D(),
        layers.Conv2D(64, kernel_size=3, padding="same"),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")
    ])
    return model

def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(64, 64, 1)),
        layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Flatten(), layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(100,))
    fake_img = generator(gan_input)
    gan_output = discriminator(fake_img)
    model = keras.Model(gan_input, gan_output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)

# Prepare training data (grayscale images)
real_images_array = np.array([cv2.resize(img.squeeze(), (64, 64)) for img in X_full])
real_images_array = np.expand_dims(real_images_array, axis=-1).astype("float32") / 255.0

# GAN Training Loop
epochs = 1000
batch_size = 32
noise_dim = 100

for epoch in tqdm(range(epochs + 1)):
    idx = np.random.randint(0, real_images_array.shape[0], batch_size)
    real_imgs = real_images_array[idx]
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    fake_imgs = generator.predict(noise)
    real_labels, fake_labels = np.ones((batch_size, 1)), np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    gan.train_on_batch(noise, np.ones((batch_size, 1)))

# Generate synthetic images
num_synthetic = 200
random_noise = np.random.normal(0, 1, (num_synthetic, 100))
synthetic_images = generator.predict(random_noise)
synthetic_images_resized = np.array([cv2.resize(img.squeeze(), (128, 128)) for img in synthetic_images])
synthetic_images_resized = np.expand_dims(synthetic_images_resized, axis=-1) / 255.0

# Assign all generated images as CANCER class (1)
synthetic_labels = np.ones(synthetic_images_resized.shape[0])  # Class 1 = Cancer

synthetic_images_scaled = (synthetic_images * 255).astype(np.uint8)
np.save("synthetic_images.npy", synthetic_images_scaled)

# Merge with real dataset (now with predicted labels)
X_imgs_total = np.concatenate([X_full, synthetic_images_resized], axis=0)
y_total = np.concatenate([y_full, synthetic_labels], axis=0)

# Optional Shuffle
indices = np.arange(X_imgs_total.shape[0])
np.random.shuffle(indices)

X_imgs_total = X_imgs_total[indices]
y_total = y_total[indices]
