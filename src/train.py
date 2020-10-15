import os
import errno
import pickle
import configparser

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from model import Predictor
from diy_functions import produce_canditates


def main():
    # Read Config Info
    config_ini = configparser.ConfigParser()
    config_filepath = './../config.ini'
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(errno.ENOENT, 
                                os.strerror(errno.ENOENT), 
                                config_filepath)
    
    config_ini.read(config_filepath, encoding='utf-8')
    config_default = config_ini['DEFAULT']

    dim_size = int(config_default.get('DIM_SIZE'))
    epoch_num = int(config_default.get('EPOCH_NUM'))
    hidden_size = int(config_default.get('HIDDEN_SIZE'))
    batch_size = int(config_default.get('BATCH_SIZE'))
    patience = int(config_default.get('PATIENCE'))

    # Load Data from pkl files
    with open(config_default.get('MIDDLE_LAYER'), 'rb') as f:
        df = pickle.load(f)

    print(df.head())

    with open(config_default.get('X_TRAIN'), 'rb') as f:
        x_train = pickle.load(f)

    with open(config_default.get('Y_TRAIN'), 'rb') as f:
        y_train = pickle.load(f)

    with open(config_default.get('X_VALID'), 'rb') as f:
        x_valid = pickle.load(f)

    with open(config_default.get('Y_VALID'), 'rb') as f:
        y_valid = pickle.load(f)

    with open(config_default.get('X_TEST'), 'rb') as f:
        x_test = pickle.load(f)

    with open(config_default.get('Y_TEST'), 'rb') as f:
        y_test = pickle.load(f)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    x_valid = torch.from_numpy(x_valid).float()
    y_valid = torch.from_numpy(y_valid).float()

    with open(config_default.get('DF_TRAIN'), 'rb') as f:
        df_train_test = pickle.load(f)

    with open(config_default.get('DF_VALID'), 'rb') as f:
        df_valid_test = pickle.load(f)

    with open(config_default.get('DF_TEST'), 'rb') as f:
        df_test = pickle.load(f)

    y_candis_train = produce_canditates(df, df_train_test, 0)
    y_candis_train = torch.from_numpy(y_candis_train).float()

    y_candis_valid = produce_canditates(df, df_valid_test, 0)
    y_candis_valid = torch.from_numpy(y_candis_valid).float()

    y_candis_test = produce_canditates(df, df_test, 0)
    y_candis_test = torch.from_numpy(y_candis_test).float()

    # Model Predict
    model = Predictor(dim_size, hidden_size, batch_size, dim_size)
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=patience)
    model.get_loader(x_train, y_train, x_valid, y_valid, x_test, y_test, 
                       y_candis_train, y_candis_valid, y_candis_test)
    
    trainer = Trainer(early_stop_callback=early_stop_callback,
                      # gpus=[0],
                      max_epochs=epoch_num)

    result = trainer.fit(model)
    print(result)

    trainer.test()


if __name__ == '__main__':
    main()

