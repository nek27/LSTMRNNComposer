from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential


class LSTMRNN(object):
	
	@staticmethod
	def build(network_input, n_vocab):
		
		model = Sequential()
		model.add(LSTM(256, 
			input_shape=(network_input.shape[1], network_input.shape[2]),
			return_sequences=True))
		model.add(Dropout(0.3))
		model.add(LSTM(512, return_sequences=True))
		model.add(Dropout(0.3))
		model.add(LSTM(256))
		model.add(Dense(256))
		model.add(Dropout(0.3))
		model.add(Dense(n_vocab))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
		
		return model
