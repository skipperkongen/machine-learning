import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
import sys

WINDOW_SIZE = 100
LOWER_CASE = True

class Vocab:

    def __init__(self, vocab_file, lower_case=False):
        with open(vocab_file) as fin:
            raw_text = fin.read()
        if lower_case:
            raw_text = raw_text.lower()
        self.distinct = sorted(list(set(raw_text)))
        self.n = len(self.distinct)
        self.int_to_char = dict([(i, c) for i, c in enumerate(self.distinct)])
        self.char_to_oh = dict([(c, np.identity(self.n)[i: i+1][0]) for i, c in enumerate(self.distinct)])

    def text_to_onehot(self, text):
        return [self.char_to_oh[c] for c in text]

    def index_to_character(self, index):
        return self.int_to_char[index]

class Agent:

    def __init__(self):
        pass

    def _get_model(self, model_name, vocab=None, window_size=WINDOW_SIZE):
        if vocab is None:
            raise Error('Vocab must be set')
        if model_name == 'micro':
            model = Sequential()
            model.add(LSTM(256, input_shape=(window_size, vocab.n)))
            model.add(Dropout(0.2))
            model.add(Dense(vocab.n, activation='softmax'))
        elif model_name == 'big':
            model = Sequential()
            model.add(LSTM(256, input_shape=(window_size, vocab.n), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(256))
            model.add(Dropout(0.2))
            model.add(Dense(vocab.n, activation='softmax'))
        else:
            raise Error('Unknown model size')

        return model

    def _get_X_y(self, raw_text, vocab=None, window_size=None, lower_case=False):
        raw_text = "".join(c for c in raw_text if c in vocab.distinct)
        if lower_case:
            raw_text = raw_text.lower()
        data_X = []
        data_y = []
        n_chars = len(raw_text)
        for i in range(0, n_chars - window_size, 1):
            seq_in = vocab.text_to_onehot(raw_text[i: i + window_size])
            seq_out = vocab.text_to_onehot(raw_text[i+window_size])
            data_X.append(seq_in)
            data_y.append(seq_out)
        X = np.reshape(data_X, (len(data_X), window_size, vocab.n))
        y = np.reshape(data_y, (len(data_X), vocab.n))
        return X, y

    def fit(self, args):
        print ('FIT:')
        print ('Model name:', args.model_name)
        print ('Initialize vocab')
        vocab = Vocab(args.vocab_file, lower_case=LOWER_CASE)
        print ('- Character set:', vocab.distinct)
        print ('Reading training data')
        with open(args.training_file) as f:
            raw_text = f.read().lower()
        print ('-', raw_text[0:25], '...')
        print ('Transforming data to X and y')
        X, y = self._get_X_y(raw_text, vocab=vocab, window_size=WINDOW_SIZE, lower_case=LOWER_CASE)
        filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        print ('Creating model')
        model = self._get_model(args.model_name, vocab=vocab)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())
        print ('Train model')
        if args.model_name == 'micro': epochs = 1
        if args.model_name == 'big': epochs = 200
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        model.fit(X, y, epochs=epochs, batch_size=128, callbacks=[checkpoint])

    def predict(self, args):
        print ('PREDICT:')
        print ('Initialize vocab')
        vocab = Vocab(args.vocab_file, lower_case=LOWER_CASE)
        print ('- Character set:', vocab.distinct)
        print ('Preprocessing seed')
        seed = args.seed
        # Truncate / pad seed
        seed = (' ' * max(0, WINDOW_SIZE - len(seed)) + seed)[:WINDOW_SIZE]
        if LOWER_CASE:
            seed = seed.lower()
        print ('- Seed:', seed)
        window = vocab.text_to_onehot(seed)
        print ('Creating model and setting model weights')
        model = self._get_model(args.model_name, vocab=vocab)
        model.load_weights(args.weights_file)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())
        print ('Generating text...')
        print ()
        print ()
        for i in range(10000):
            prediction = model.predict(np.reshape(window, (1, WINDOW_SIZE, vocab.n)), verbose=0)
            index = np.argmax(prediction)
            sys.stdout.write(vocab.index_to_character(index))
            sys.stdout.flush()
            window.append(np.identity(vocab.n)[index: index+1][0])
            window = window[1:]

if __name__=='__main__':

    agent = Agent()

    parser = argparse.ArgumentParser(prog='AGENT')
    parser.add_argument('-m', '--model-name', default='big', choices=['micro', 'big'], help='Size of model')
    parser.add_argument('-v', '--vocab-file', default='vocab-alice.txt', metavar='TEXT FILE', help='Vocabulary file that contains the recognized character set')
    subparsers = parser.add_subparsers(help='sub-command help')
    # Fit command
    parser_a = subparsers.add_parser('fit', help='a help')
    parser_a.add_argument('--training-file', default='alice.txt', help='Path to text file with training data. Only characters in vocab file used.')
    parser_a.set_defaults(func=agent.fit)
    # Predict command
    parser_b = subparsers.add_parser('predict', help='b help')
    parser_b.add_argument('-s', '--seed', metavar='SEED', default='the quick brown fox jumped over the lazy dog.', help='An input string to use as seed')
    parser_b.add_argument('-w', '--weights-file', default='alice-big-W100-L0.2582.hdf5', help='path to hdf5 file that contains weights')
    parser_b.set_defaults(func=agent.predict)

    args = parser.parse_args()
    args.func(args)
