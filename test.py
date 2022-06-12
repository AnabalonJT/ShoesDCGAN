import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot


def save_plot(examples, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])
    filename = "fake.png"
    pyplot.savefig(filename)
    pyplot.close()


if __name__ == "__main__":
    model = load_model("saved_model/g_model_0.h5")

    n_samples = 25
    latent_dim = 128
    latent_points = np.random.normal(size=(n_samples, latent_dim))
    examples = model.predict(latent_points)
    save_plot(examples, int(np.sqrt(n_samples)))