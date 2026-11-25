import pandas as pd
import tensorflow as tf
from src.pipeline import IMG_SIZE, load_image, make_dataset


def test_load_image_shape_and_range(tmp_path):
    raw = tf.cast(tf.random.uniform((50, 40, 1), 0, 256, dtype=tf.int32), tf.uint8)
    path = tmp_path / "img.png"
    tf.io.write_file(str(path), tf.io.encode_png(raw))

    img = load_image(str(path))
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert img.dtype == tf.float32
    assert float(tf.reduce_min(img)) >= 0.0
    assert float(tf.reduce_max(img)) <= 255.0


def test_make_dataset_preserves_order(tmp_path):
    paths = []
    for i in range(3):
        raw = tf.fill((32, 32, 1), tf.cast(i * 50, tf.uint8))
        p = tmp_path / f"{i}.png"
        tf.io.write_file(str(p), tf.io.encode_png(raw))
        paths.append(str(p))

    df = pd.DataFrame({"path": paths, "label": [1, 0, 1]})
    ds = make_dataset(df, lambda x: x, batch_size=32, shuffle=False)
    imgs, labels = next(iter(ds))
    assert imgs.shape == (3, IMG_SIZE, IMG_SIZE, 3)
    assert labels.numpy().tolist() == df["label"].tolist()
