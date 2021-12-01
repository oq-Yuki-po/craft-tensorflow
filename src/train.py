import datetime
import pathlib
import shutil

import tensorflow as tf
import typer
from tensorflow.keras import optimizers

from src.callbacks import (CheckLearningProcess, CustomLearningRateScheduler, CustomModelCheckpoint, cb_early_stopping,
                           cb_epoch_checkpoint, cb_tensorboard)
from src.dataset.generator import CraftDataset
from src.loss import CustomLoss
from src.model import craft
from src.util import load_yaml


def train():

    cfg = load_yaml()

    result_dir = f"results/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    checkpoint_dir = f"{result_dir}/checkpoints/"
    log_dir = f"{result_dir}/logs"
    image_dir = f"{result_dir}/images"

    pathlib.Path(result_dir).mkdir(exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)
    pathlib.Path(log_dir).mkdir(exist_ok=True)
    pathlib.Path(image_dir).mkdir(exist_ok=True)
    pathlib.Path(f"{image_dir}/region").mkdir(exist_ok=True)
    pathlib.Path(f"{image_dir}/affinity").mkdir(exist_ok=True)

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    model = craft()

    if cfg['is_model_weight_load']:

        latest = tf.train.latest_checkpoint(cfg['train_checkpoint'])
        model.load_weights(latest)
        print(f'weight loaded : {latest}')

    optimizer = optimizers.Adam()

    batch_size = cfg['train_batch_size']

    if cfg['is_weak_supervised']:
        craft_dataset_synth = CraftDataset()
        train_ds_synth = craft_dataset_synth.generate()
        craft_dataset_icdar = CraftDataset(model=model)
        train_ds_icdar = craft_dataset_icdar.generate(is_weak_supervised=cfg['is_weak_supervised'])
        steps_per_epoch = (cfg['train_synth_data_length'] + cfg['train_icdar_data_length']
                           ) // (cfg['train_synth_batch_size'] + cfg['train_icdar_batch_size']) + 1

        train_ds_synth = train_ds_synth.\
            shuffle(buffer_size=cfg['train_shuffle_buffer_size'], seed=cfg['train_shuffle_seed']).\
            repeat().\
            batch(cfg['train_synth_batch_size']).\
            prefetch(tf.data.AUTOTUNE)

        train_ds_icdar = train_ds_icdar.\
            shuffle(buffer_size=cfg['train_shuffle_buffer_size'], seed=cfg['train_shuffle_seed']).\
            repeat().\
            batch(cfg['train_icdar_batch_size']).\
            prefetch(tf.data.AUTOTUNE)

        train_ds = tf.data.Dataset.zip((train_ds_synth, train_ds_icdar))

        train_ds = train_ds.map(craft_dataset_synth.concat_dataset)
    else:
        craft_dataset = CraftDataset()
        train_ds = craft_dataset.generate()
        steps_per_epoch = cfg['train_synth_data_length'] // batch_size + 1

        train_ds = train_ds.\
            repeat().\
            batch(batch_size).\
            prefetch(tf.data.AUTOTUNE)

    model.compile(optimizer=optimizer, loss=CustomLoss(batch_size), run_eagerly=True)

    model.optimizer.lr.assign(cfg['train_initial_lr'])

    model.summary()

    print(f'steps_per_epoch: {steps_per_epoch}')

    shutil.copy("src/config.yml", f"{result_dir}/config.yaml")

    callbacks = [cb_tensorboard(log_dir),
                 cb_epoch_checkpoint(checkpoint_dir),
                 cb_early_stopping(),
                 CustomLearningRateScheduler(change_steps=cfg['train_lr_change_step']),
                 CheckLearningProcess(image_dir),
                 CustomModelCheckpoint(model, checkpoint_dir, save_steps=cfg['train_save_steps'])]

    model.fit(train_ds,
              epochs=cfg['train_epochs'],
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)

    model.save(f'{result_dir}/saved_model', include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    open(f"{result_dir}/model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    typer.run(train)
