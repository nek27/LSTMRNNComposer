# coding=utf8
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop


class LSTMRNN(object):
	
	@staticmethod
	def build(network_input, n_vocab):
		
		model = Sequential()
		#model.add(LSTM(256, 
		model.add(LSTM(512,
			input_shape=(network_input.shape[1], network_input.shape[2]),
			return_sequences=True))
		model.add(Dropout(0.3))
		model.add(LSTM(512, return_sequences=True))
		model.add(Dropout(0.3))
		#model.add(LSTM(256))
		model.add(LSTM(512))
		model.add(Dense(256))
		model.add(Dropout(0.3))
		model.add(Dense(n_vocab))
		model.add(Activation('softmax'))
		
		opt = RMSprop(lr = 0.0001)
		model.compile(loss='categorical_crossentropy', optimizer = opt)
		
		return model
