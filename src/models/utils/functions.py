import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import utils.consts



def read_and_resize_img(img_path, img_shape):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_shape[0], img_shape[1]])
    return image

def read_possible_genres():
    possible_genres = open(f'{utils.consts.DATASET_PATH}/possible_genres.txt', 'r').read().split('\n')
    possible_genres.pop()
    return possible_genres

def load_arts_dataset():
    return pd.read_csv(utils.consts.ARTS_PATH)

def load_train_generator(df_arts, img_shape):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

    train_df, _ = train_test_split(df_arts, test_size=0.05, stratify=df_arts['artist id'], random_state=utils.consts.SEED)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=utils.consts.IMAGES_PATH,
        x_col='image path',
        y_col=read_possible_genres(),
        batch_size=32,
        seed=utils.consts.SEED,
        shuffle=True,
        class_mode="raw",
        target_size=(img_shape[0], img_shape[1])
    )
    
    return train_generator

def load_test_dataset(df_arts):
    return train_test_split(df_arts, test_size=0.05, stratify=df_arts['artist id'], random_state=utils.consts.SEED)[1]
