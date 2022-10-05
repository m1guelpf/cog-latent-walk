import keras_cv
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from utils import export_as_gif
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        # keras.mixed_precision.set_global_policy("mixed_float16")
        self.model = keras_cv.models.StableDiffusion(
            img_height=512, img_width=512, jit_compile=True
        )

    def predict(
        self,
        prompt_1: str = Input(description="Input 1 prompt", default=""),
        prompt_2: str = Input(description="Input 2 prompt", default=""),
        interpolation_steps: int = Input(
            description="Interpolation steps",
            ge=1,
            default=50,
        ),
    ) -> Path:
        encoding_1 = tf.squeeze(self.model.encode_text(prompt_1))
        encoding_2 = tf.squeeze(self.model.encode_text(prompt_2))

        # This value might to be lowered to 3 on a smaller GPU.
        batch_size = 10
        batches = interpolation_steps // batch_size

        interpolated_encodings = tf.linspace(
            encoding_1, encoding_2, interpolation_steps
        )
        batched_encodings = tf.split(interpolated_encodings, batches)

        seed = 12345
        noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

        images = []
        for batch in range(batches):
            images += [
                Image.fromarray(img)
                for img in self.model.generate_image(
                    batched_encodings[batch],
                    batch_size=batch_size,
                    num_steps=25,
                    diffusion_noise=noise,
                )
            ]

        export_as_gif("/tmp/out.gif", images, rubber_band=True)
        return Path("/tmp/out.gif")
