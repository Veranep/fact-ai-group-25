from pandas import read_csv

import numpy as np
from math import floor
import pdb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from typing import Dict, Any

class EmbeddingModel(pl.LightningModule):

    def __init__(self, num_inputs, num_embed, num_hidden, num_outputs):
        super().__init__()
        self.save_hyperparameters()
        self.embed = nn.Embedding(num_inputs, num_embed)
        self.linear1 = nn.Linear(2*num_embed, num_hidden)
        self.act_fn = nn.ELU()
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.loss_module = nn.MSELoss()

    def forward(self, x):
        origin_embed =  self.embed(x[:,0])
        destination_embed = self.embed(x[:,1])
        state_embed = torch.hstack((origin_embed, destination_embed))
        state_embed = self.linear1(state_embed)
        state_embed = self.act_fn(state_embed)
        state_embed = self.linear2(state_embed)
        return state_embed
    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001,
                                     betas=(0.9, 0.999), eps=1e-07,
                                     amsgrad=False)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        loss = self.loss_module(preds, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        loss = self.loss_module(preds, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        loss = self.loss_module(preds, target)
        self.log('test_loss', loss, on_step=False, on_epoch=True)


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    # Get Travel Times
    travel_times = read_csv('../../data/ny/zone_traveltime.csv', header=None).values
    mean_val = np.mean(travel_times)
    max_val = np.abs(travel_times).max()
    print("Mean: {}, Max: {}".format(mean_val, max_val))
    travel_times -= mean_val
    travel_times /= max_val

    # Define NN
    model = EmbeddingModel(travel_times.shape[0] + 1, 10, 100, 1)

    # Format
    X: Dict[str, Any] = {'origin_input': [], 'destination_input': []}
    y = []
    for origin in range(travel_times.shape[0]):
        for destination in range(travel_times.shape[1]):
            X['origin_input'].append(origin + 1)
            X['destination_input'].append(destination + 1)
            y.append(travel_times[origin, destination])

    # Get train/test split
    idxs = np.array(list(range(len(y))))
    np.random.shuffle(idxs)

    train_idxs = idxs[0:floor(0.8 * len(y))]
    valid_idxs = idxs[floor(0.8 * len(y)) + 1:floor(0.9 * len(y))]
    test_idxs = idxs[floor(0.9 * len(y)) + 1:]

    X_train = {key: np.array(value)[train_idxs] for key, value in X.items()}
    X_train = torch.tensor(np.array([value for value in X_train.values()]).T,
                           dtype=torch.int32)
    X_valid = {key: np.array(value)[valid_idxs] for key, value in X.items()}
    X_valid = torch.tensor(np.array([value for value in X_valid.values()]).T,
                           dtype=torch.int32)
    X_test = {key: np.array(value)[test_idxs] for key, value in X.items()}
    X_test = torch.tensor(np.array([value for value in X_test.values()]).T,
                          dtype=torch.int32)
    y_train = (np.array(y)[train_idxs]).reshape((-1, 1))
    y_train = torch.Tensor(y_train)
    y_valid = (np.array(y)[valid_idxs]).reshape((-1, 1))
    y_valid = torch.Tensor(y_valid)
    y_test = (np.array(y)[test_idxs]).reshape((-1, 1))
    y_test = torch.Tensor(y_test)
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 1024,
                              pin_memory=True, num_workers=3)
    val_loader = DataLoader(valid_dataset, shuffle=False, batch_size = 1024,
                              drop_last=False, num_workers=3)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size = 1024,
                             drop_last=False, num_workers=3)

    trainer = pl.Trainer(default_root_dir='../../models/embedding.h5',
                         deterministic = True,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=1000,
                         callbacks=[EarlyStopping(monitor="val_loss",
                                                   patience=15),
                                    ModelCheckpoint(mode="min",
                                                    monitor="val_loss")])
    trainer.logger._log_graph = True
    # Train
    trainer.fit(model, train_loader, val_loader)

    model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_loss = trainer.test(model, test_dataloaders=test_loader, verbose=True)
    print("Loss on test fraction: {}".format(test_loss))

    # Save Embeddings
    pickle.dump(model.embed.weight, open('../../data/ny/embedding_weights.pkl', 'wb'))
