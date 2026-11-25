"""Pipeline tf.data para carregamento, aumento de dados e pré-processamento de radiografias."""
from collections.abc import Callable

import keras
import pandas as pd
import tensorflow as tf

from src.config import SEED

IMG_SIZE: int = 224

AUGMENT: keras.Sequential = keras.Sequential(
    [
        keras.layers.RandomRotation(0.03, seed=SEED),
        keras.layers.RandomZoom(0.1, seed=SEED),
        keras.layers.RandomContrast(0.1, seed=SEED),
    ],
    name="augment",
)


def load_image(path: str) -> tf.Tensor:
    """Lê PNG em escala de cinza, redimensiona para 224x224 e devolve float32 (224, 224, 3) em [0, 255]."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.clip_by_value(tf.cast(img, tf.float32), 0.0, 255.0)
    img = tf.image.grayscale_to_rgb(img)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


def make_dataset(
    df: pd.DataFrame,
    preprocess: Callable,
    batch_size: int = 32,
    shuffle: bool = False,
    augment: keras.Sequential | None = None,
) -> tf.data.Dataset:
    """Monta tf.data.Dataset na ordem map(load_image) -> augment -> preprocess -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].astype(str).tolist(), df["label"].astype("int32").tolist())
    )
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
