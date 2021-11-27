import datetime
import pathlib
import shutil

import numpy as np
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

    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

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

        latest = tf.train.latest_checkpoint('results/checkpoints')
        model.load_weights(latest)
        print(f'weight loaded : {latest}')

    optimizer = optimizers.Adam(learning_rate=cfg['train_initial_lr'])

    batch_size = cfg['train_batch_size']

    if cfg['is_weak_supervised']:
        craft_dataset_synth = CraftDataset()
        train_ds_synth = craft_dataset_synth.generate()
        craft_dataset_icdar = CraftDataset(model=model)
        train_ds_icdar = craft_dataset_icdar.generate(is_weak_supervised=cfg['is_weak_supervised'])
        steps_per_epoch = (cfg['train_synth_data_length'] + cfg['train_icdar_data_length']) // batch_size + 1

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
    else:
        craft_dataset = CraftDataset()
        train_ds = craft_dataset.generate()
        steps_per_epoch = cfg['train_synth_data_length'] // batch_size + 1

        train_ds = train_ds.\
            repeat().\
            batch(batch_size).\
            prefetch(tf.data.AUTOTUNE)

    model.compile(optimizer=optimizer, loss=CustomLoss(batch_size), run_eagerly=True)

    # model.summary()

    shutil.copy("src/config.yml", f"{result_dir}/config.yaml")

    callbacks = [cb_tensorboard(log_dir),
                #  cb_epoch_checkpoint(checkpoint_dir),
                 cb_early_stopping(),
                 CustomLearningRateScheduler(change_steps=cfg['train_lr_change_step']),
                 CheckLearningProcess(image_dir),
                 CustomModelCheckpoint(model, checkpoint_dir, all_step=0, save_steps=cfg['train_save_steps'])]

    callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=model)

    if cfg['is_weak_supervised']:
        logs = {}
        train_summary_writer = tf.summary.create_file_writer(log_dir)
        callbacks.on_train_begin(logs=logs)
        loss_fn = CustomLoss(batch_size)
        epoch = 1
        callbacks.on_epoch_begin(epoch)
        for step, (synth, icdar) in enumerate(zip(train_ds_synth, train_ds_icdar), start=1):
            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_train_batch_begin(step, logs=logs)
            x_synth, y_synth = synth[0], synth[1]
            x_icdar, y_icdar = icdar[0], icdar[1]
            x_batch = np.concatenate([x_synth.numpy(), x_icdar.numpy()])
            y_batch = np.concatenate([y_synth.numpy(), y_icdar.numpy()])
            loss_value = train_step(x_batch, y_batch)

            if step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value, step=step)
            print("Training loss (for one batch) at step %d: %.4f on epoch %d" % (step, float(loss_value), epoch))
            print("Seen so far: %d samples" % ((step + 1) * batch_size))
            callbacks.on_train_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)

            if step % steps_per_epoch == 0:
                callbacks.on_epoch_end(epoch, logs=logs)
                epoch += 1
                callbacks.on_epoch_begin(epoch)

            if step == cfg['train_end_step']:
                break

        callbacks.on_train_end(logs=logs)

    else:
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
