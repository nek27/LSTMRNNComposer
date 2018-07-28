# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint
from utils import *
from model import LSTMRNN


notes = read_dataset('midi_songs')

rnn_input, rnn_output, n_vocab = create_inp_out(notes)

model = LSTMRNN.build(rnn_input, n_vocab)

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    

checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)
callbacks_list = [checkpoint]

model.fit(rnn_input, rnn_output, epochs=50, batch_size=64, callbacks=callbacks_list)
