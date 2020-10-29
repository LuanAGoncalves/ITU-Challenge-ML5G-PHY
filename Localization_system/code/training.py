import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import metrics
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta, Adam
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os

from models import build_default_mlp


def prepare_dataset(dataset_folder):
    channels = np.load(
        dataset_folder + "mimoChannels/mimoChannels_train.npz"
        if dataset_folder[-1] == "/"
        else dataset_folder + "/mimoChannels/mimoChannels_train.npz"
    )["output_classification"]

    coord = np.load(
        dataset_folder + "coord_train.npz"
        if dataset_folder[-1] == "/"
        else dataset_folder + "/coord_train.npz"
    )["coordinates"]

    context = np.load(
        dataset_folder + "context_train.npz"
        if dataset_folder[-1] == "/"
        else dataset_folder + "/context_train.npz"
    )["context"]

    idx = context == 1

    channels = channels[idx]
    coord = coord[idx]

    n, _, _ = channels.shape
    channels = channels.reshape((n, -1))
    channels -= channels.mean(axis=0)
    channels /= channels.std(axis=0)

    return channels, coord


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", help="Learing rate", type=float, default=0.001, required=False
    )
    parser.add_argument(
        "--epochs", help="Number of epochs", type=int, default=50, required=False
    )
    parser.add_argument(
        "--seed", help="Random seed", type=int, default=2020, required=False
    )
    parser.add_argument(
        "--batch_size", help="Batch size", type=int, default=64, required=False
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        help="GPU memory fraction",
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--dataset",
        help="Dataset folder",
        type=str,
        default="../../../datasets/Raymobtime_datasets/s008/",
        required=False,
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction
    )
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    if os.path.isdir("./models"):
        pass
    else:
        os.mkdir("./models")

    channels, coords = prepare_dataset(args.dataset)

    model = build_default_mlp(256)
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse", metrics=["mae"],
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10
    )
    print(channels.min(), channels.max())

    # checkpoint = ModelCheckpoint(
    #     './models/model.h5',
    #     monitor='val_dice_coef',
    #     verbose=0,
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode='auto'
    # )

    model.fit(
        x=channels,
        y=coords,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[reduce_lr],
        validation_split=0.2,
        shuffle=True,
        initial_epoch=0,
        workers=1,
    )

    preds = model.predict(channels)
    exam = np.hstack((coords, preds))
    print(exam[:5, :])
