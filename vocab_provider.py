
import numpy as np

class VocabProvider(object):
    """
    Handles reading and writing of vocabulary files
    """
    def __init__(self, vocab_path, csv_path=None ):
        _PAD = "_PAD"
        _SEP = "_SEP"
        _UNK = "_UNK"
        _START_VOCAB = [_PAD, _SEP, _UNK]

        self.PAD_ID = 0
        self.SEP_ID = 1
        self.UNK_ID = 2

        if csv_path:
            with open(csv_path,'r') as in_file:
                text = in_file.read().lower()
            
            self.chars = _START_VOCAB + sorted(list(set(text) - set([',','\n','\r'])))
            with open(vocab_path, 'w') as out_file:
                out_file.write('\n'.join(self.chars))
        else:
            with open(vocab_path,'r') as in_file:
                text = in_file.read().lower()
            self.chars = text.splitlines()

        self.vocab = {c:i for i,c in enumerate(self.chars)}
        print('Vocabulary size: ', len(self.vocab))

    def vocab_size(self):
        return len(self.vocab)

