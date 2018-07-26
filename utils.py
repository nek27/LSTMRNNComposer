# coding=utf8
from music21 import converter, instrument, note, chord
import numpy as np
import pickle
import keras
import os


def read_dataset(folder_path):
	notes = []
	file_list = os.listdir(folder_path)
	count = 0
	
	for i, f in enumerate(file_list):
		print('Parsing {}/{} {}'.format(i, len(file_list), f))
		midi = converter.parse('{}/{}'.format(folder_path, f))
		notes_to_parse = None
		
		try: # file has instrument parts
			parts = instrument.partitionByInstrument(midi)
			notes_to_parse = parts.parts[0].recurse()
		except: # file has notes in a flat structure
			notes_to_parse = midi.flat.notes

		for element in notes_to_parse:
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))
		
		#if(count >= 2):
		#	break
		count += 1
		
	with open('data/notes', 'wb') as filepath:
		pickle.dump(notes, filepath)
        
	return notes
	
#def create_inp_out(notes : list, sequence_length = 100):
def create_inp_out(notes : list, sequence_length = 20):
	
	# Get all pitch names
	pitch_names = sorted(set(note for note in notes))
	n_vocab = len(pitch_names)
	
	# Pitch to int mapping
	note_to_int = dict( (note, number) for number, note in enumerate(pitch_names) )
	
	rnn_input = []
	rnn_output = []
	
	# Create input sequences together with its output
	for i in range(0, len(notes) - sequence_length, 1):
		seq_in = notes[i : i + sequence_length]
		seq_out = notes[i + sequence_length]
		rnn_input.append([note_to_int[note] for note in seq_in])
		rnn_output.append(note_to_int[seq_out])
		
	n_patterns = len(rnn_input)
	
	# Reshape input to be compatible with LSTM Layers
	rnn_input = np.reshape(rnn_input, (n_patterns, sequence_length, 1))
	# Normalize input
	rnn_input = rnn_input / float(n_vocab)
	
	rnn_output = keras.utils.to_categorical(rnn_output, int(n_vocab))
	
	return rnn_input, rnn_output, n_vocab
