# -*- coding: utf-8 -*-
import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from model import LSTMRNN

def compose():
	""" Compose a piano midi file """
	# Load notes used to train the RNN
	with open('data/notes', 'rb') as notesfile:
		notes = pickle.load(notesfile)
		
	# Get pitch names and amount of notes
	pitchnames = sorted(set(note for note in notes))
	n_vocab = len(pitchnames)
	
	# Get inputs
	rnn_in, normalized_rnn_in = prepare_sequences(notes, pitchnames, n_vocab)
	
	# Load model
	model = LSTMRNN.build(rnn_in, n_vocab)
	model.load_weights('weights.hdf5')
	
	# Get predictions from rnn
	pred_output = compose_pred(model, rnn_in, pitchnames, n_vocab)
	
	# Save midi file with composition
	save_midi(pred_output)
	
#def prepare_sequences(notes: list, pitchnames: list, n_vocab: int, sequence_length = 100):
def prepare_sequences(notes: list, pitchnames: list, n_vocab: int, sequence_length = 20):
	""" Prepare RNN input """
	note_to_int = dict( (note, number) for number, note in enumerate(pitchnames) )
	
	rnn_in = []
	rnn_out = []
	
	for i in range(0, len(notes) - sequence_length, 1):
		seq_in = notes[i : i + sequence_length]
		seq_out = notes[i + sequence_length]
		rnn_in.append([note_to_int[note] for note in seq_in])
		rnn_out.append(note_to_int[seq_out])
		
	n_patterns = len(rnn_in)
	
	# Reshape input to format compatible with LSTM layers and normalize
	rnn_in = np.reshape(rnn_in, (n_patterns, sequence_length, 1))
	normalized_rnn_in = rnn_in / float(n_vocab)
	
	return rnn_in, normalized_rnn_in
	
def compose_pred(model, rnn_in, pitchnames, n_vocab, n_notes = 500):
	""" Compose a new midi based on the dataset """
	# Pick random index of sequence as starting point
	start = np.random.randint(0, len(rnn_in)-1)
	int_to_note = dict( (number, note) for number, note in enumerate(pitchnames) )
	
	pattern = rnn_in[start]
	pred_output = []
	
	# Generate notes
	for note_index in range(n_notes):
		#print('{}/{}'.format(note_index, n_notes), end='\r')
		print('{}/{}'.format(note_index, n_notes))
		pred_in = np.reshape(pattern, (1, len(pattern), 1))
		pred_in = pred_in / float(n_vocab)
		
		prediction = model.predict(pred_in, verbose=0)
		
		index = np.argmax(prediction)
		result = int_to_note[index]
		pred_output.append(result)
		print(pred_output)
		
		#pattern.append(index)
		pattern = np.append(pattern, index)
		pattern = pattern[1:len(pattern)]
		
	print('done!')
	return pred_output
	
def save_midi(pred_output):
	""" Save midi file with the composotion created """
	offset = 0
	output_notes = []
	
	for pattern in pred_output:
		# Chord
		if('.' in pattern or pattern.isdigit()):
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				new_note = note.Note(int(current_note))
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			new_chord.offset = offset
			output_notes.append(new_chord)
		
		# Note
		else:
			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			output_notes.append(new_note)
			
		# Increase offset with each iter
		offset += 0.5
		
	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp='test_output.mid')
	
if __name__ == '__main__':
	compose()
		
