import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils.consts

df_artists = pd.read_csv(utils.consts.ARTISTS_PATH)

possible_genres = open(utils.consts.POSSIBLE_GENRES_PATH, 'r').read().split('\n')
possible_genres.pop()

df_arts = pd.read_csv(utils.consts.ARTS_PATH)

train_df, test_df = train_test_split(df_arts, test_size=0.05, stratify=df_arts['artist id'], random_state=utils.consts.SEED)



def read_and_resize_img(img_path, img_shape):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_shape[0], img_shape[1]])
    return image

def load_artists_dataset():
    return df_artists

def load_possible_genres():
    return possible_genres.copy()

def load_arts_dataset():
    return df_arts

def load_train_full_generator(img_shape):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_arts,
        directory=utils.consts.IMAGES_PATH,
        x_col='image path',
        y_col=possible_genres,
        batch_size=32,
        seed=utils.consts.SEED,
        shuffle=True,
        class_mode="raw",
        target_size=(img_shape[0], img_shape[1])
    )
    return train_generator

def load_train_for_test_generator(img_shape):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=utils.consts.IMAGES_PATH,
        x_col='image path',
        y_col=load_possible_genres(),
        batch_size=32,
        seed=utils.consts.SEED,
        shuffle=True,
        class_mode="raw",
        target_size=(img_shape[0], img_shape[1])
    )
    
    return train_generator

def load_train_dataset():
    return train_df

def load_test_dataset():
    return test_df


def read_and_resize_img(img_path, img_shape=(256, 256, 3)):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_shape[0], img_shape[1]])
    return image

def show_image(img_path, img_shape=(256, 256, 3)):
    plt.imshow(read_and_resize_img(img_path, img_shape))
    plt.axis('off')

def show_top_10_images(similar_images_df, img_shape=(256, 256, 3)):
    fig, axs = plt.subplots(2, 5, figsize=(16, 7))

    imgs_path = similar_images_df['image path'].iloc[:10].values

    for i in range(2):
        for j in range(5):
            axs[i, j].set_title(f'Top {i*5+j+1}')
            axs[i, j].imshow(read_and_resize_img(imgs_path[i*5+j], img_shape))
            axs[i, j].axis('off')
