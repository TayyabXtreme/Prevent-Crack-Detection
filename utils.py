"""Model loading and preprocessing. Preprocessing mirrors `load_image` in the training notebook."""
import io
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()   # iPhone HEIC/HEIF photos
except ImportError:
    pass

MODEL_DIR = Path(__file__).parent / "model"
DEMO_DIR = Path(__file__).parent / "demo_images"
MODEL_PATH = MODEL_DIR / "crack_model.keras"
INFO_PATH = MODEL_DIR / "model_info.json"

DEFAULT_INFO = {"model_name": "unknown", "class_names": ["Negative", "Positive"], "img_size": 224, "threshold": 0.5}


@keras.saving.register_keras_serializable(package="crack")
class PositionEmbedding(layers.Layer):
    """Same custom layer as the notebook's ViT; must be registered so the ViT can be loaded."""
    def __init__(self, num_patches, dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches, self.dim = num_patches, dim

    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(1, self.num_patches, self.dim),
                                   initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos

    def get_config(self):
        return {**super().get_config(), "num_patches": self.num_patches, "dim": self.dim}


def load_info():
    if INFO_PATH.exists():
        return {**DEFAULT_INFO, **json.loads(INFO_PATH.read_text())}
    return DEFAULT_INFO


def load_model():
    return keras.models.load_model(MODEL_PATH)


def to_rgb(image: Image.Image) -> Image.Image:
    """Any mode (grayscale, palette, CMYK, 16-bit, alpha...) -> upright 8-bit RGB, the format the model was trained on."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):   # 16-bit / float images: stretch to 0-255
        arr = np.asarray(image, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        image = Image.fromarray(((arr - lo) / (hi - lo) * 255 if hi > lo else arr * 0).astype(np.uint8))
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))   # transparent areas -> white
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def open_image(data: bytes):
    """Bytes of any image format -> (RGB image, description), or (None, error) if it is not a readable image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()   # first frame for GIF / multi-page TIFF
        desc = f"{image.format or 'image'} · {image.mode} · {image.width}×{image.height}"
        return to_rgb(image), desc
    except Exception as e:
        return None, str(e) or type(e).__name__


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """PIL image -> (size, size, 3) float32 array in 0-255, exactly like training."""
    image = to_rgb(image)
    x = tf.image.resize(np.asarray(image), (size, size))
    return tf.cast(tf.saturate_cast(x, tf.uint8), tf.float32).numpy()


def predict(model, images: list[Image.Image], size: int) -> np.ndarray:
    """Returns P(crack) for each image."""
    batch = np.stack([preprocess(img, size) for img in images])
    return model.predict(batch, verbose=0).ravel()
