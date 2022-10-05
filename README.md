# SD Latent Walk Cog model

This is an implementation of the [Stable Diffusion Latent Walk](https://keras.io/examples/generative/random_walks_with_stable_diffusion/) colab as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

Then, you can run predictions:

    cog predict -i prompt1="Mysterious trail in the snow, concept art, digital, artstation" -i prompt2="Mysterious trail in the forest on a summer morning, concept art, digital, artstation"

Or, build a Docker image:

    cog build

Or, [push it to Replicate](https://replicate.com/docs/guides/push-a-model):

    cog push r8.im/...
