
import numpy as np
import random

from vocab_provider import VocabProvider
from heuristic import get_overlaps


class DataProvider(object):
    def __init__(self, csv_path, vocab_provider, partitions=None, shuffle=False, add_dominance=False):

        assert isinstance(vocab_provider, VocabProvider), 'vocab_provider is not a VocabProvider class'
        self.vocab_provider = vocab_provider
        self.vocab = vocab_provider.vocab

        with open(csv_path,'r') as in_file:
            text = in_file.read().lower()
        prep_data, self.maxlen_source, self.maxlen_target = self.prepare_data(text)

        if shuffle:
            random.shuffle(prep_data)

        self.p1, self.p2 = self.estimate_target_distributions(prep_data, self.get_train_size(prep_data, partitions), self.maxlen_source)
        self.full, self.sources, self.targets = self.convert_data(prep_data, self.maxlen_source, vocab_provider, self.p1, self.p2, add_dominance)

        if partitions:
            assert isinstance(partitions, list), 'partitions must be a list of tuples'
            self.data_set = self.partition_data(self.full, partitions)
        else:
            self.data_set = {}
            


    def prepare_data(self, raw):

        print('Prepare data...')
        
        def true_count(items):
            return sum([int(i) for i in items])
        def binary_str(start, end, length):
            return [int(i >= start and i < end) for i in range(length)]

        lines2 = []
        maxlen_source = 0
        maxlen_target = 0

        for line in raw.splitlines():
            words = line.split(',')
            target, w1, w2 = words
            tl, l1, l2 = [len(w) for w in words]
            
            left = [target.startswith(w1[:k+1]) for k in range(l1)]
            right = [target.endswith(w2[k:]) for k in range(l2)]
            
            lc, rc = true_count(left), true_count(right)
            if lc > 0 and rc > 0 and (lc + rc >= tl):
                lf = lc/tl
                rf = rc/tl
                li = int(lf >= 0.5)
                ri = int(rf >= 0.5)
                
                if abs(lf - rf) <= 0.1:
                    li = ri = 1
                    
                if lc + rc > tl:
                    if lc <= rc:
                        rc = tl - lc
                    else:
                        lc = tl - rc
                
                i1 = lc - 1
                i2 = l2 - rc

                binary = binary_str(0, i1 + 1, l1) + [0] + binary_str(i2, l2, l2)
                assert target == w1[:i1+1] + w2[i2:], 'Mismatch: ' + w1[:i1+1] + w2[i2:]

            
                if maxlen_source < l1 + l2 + 1:
                    maxlen_source = l1 + l2 + 1

                if maxlen_target < tl:
                    maxlen_target = tl

                lines2.append((w1, w2, target, li, ri, binary))
        
        print('Total examples: ', len(lines2))
        print('Max len source: ', maxlen_source)
        print('Max len target: ', maxlen_target)
        return lines2, maxlen_source, maxlen_target
    
    def convert_data(self, prep_data, maxlen_source, vocab_provider, p1, p2, add_dominance):
        print('Converting inputs...')
        vocab = vocab_provider.vocab
        vocab_size = len(vocab)
        data = []
        sources = []
        targets = []
        
        ad = int(add_dominance)

        for line in prep_data:
            w1, w2, target, li, ri, b = line
            x = (list(map(lambda s: vocab.get(s, vocab_provider.UNK_ID) + (ad * li * vocab_size), w1)) + 
                [vocab_provider.SEP_ID] + 
                list(map(lambda s: vocab.get(s, vocab_provider.UNK_ID) + (ad * ri * vocab_size), w2)))
            
            v = get_overlaps(w1, w2, p1, p2)
            assert len(v) == len(w1) + len(w2) + 1, "Length mismatch. Should be %d but got %d " % (len(w1) + len(w2) + 1, len(v)) 

            x.extend([0]*(maxlen_source - len(x)))
            b.extend([0]*(maxlen_source - len(b)))
            v.extend([[0]]*(maxlen_source - len(v)))
            assert len(x) == maxlen_source, "Exceeds max source length: " + w1
            
            sources.append(w1 + ' ' + w2)
            targets.append(target)
            data.append([x, v, b])
        
        return data, sources, targets

    def partition_data(self, full, partitions):
        print('Partitions:')
        full_size = len(full)
        last_size = 0
        data_set = {}
        for name, frac in partitions:
            set_size = int(full_size * frac)
            set_data = full[last_size:last_size + set_size]
            data_set[name] = (set_data, last_size)
            print('%s set size: %d' % (name, len(set_data)))
            last_size += set_size
        
        return data_set

    def get_train_size(self, data, partitions):
        train_size = len(data)
        if partitions:
            assert isinstance(partitions, list), 'partitions must be a list of tuples'
            for name, frac in partitions:
                if name is 'train':
                    train_size = int(len(data) * frac)
                    break

        return train_size

    def estimate_target_distributions(self, prep_data, train_size, maxlen_source):
        p1 = np.ones([maxlen_source, maxlen_source]) * 1e-6
        p2 = np.ones([maxlen_source, maxlen_source]) * 1e-6

        for w1, w2, _, i1, i2, _ in prep_data[:train_size]:
            l1 = len(w1) - 1
            l2 = len(w2) - 1
            p1[l1, i1] += 1
            p2[l2, i2] += 1

        p1 = p1 / np.sum(p1, axis=1).reshape([maxlen_source, 1])
        p2 = p2 / np.sum(p2, axis=1).reshape([maxlen_source, 1])

        return p1, p2

    def map_inputs(self, w1, w2, flip=True, d1=0, d2=0):
        maxlen = self.maxlen_source
        vocab_size = len(self.vocab)

        inputs = list(map(lambda s: self.vocab.get(s, self.vocab_provider.UNK_ID) + vocab_size * d1, w1.lower())) + [self.vocab_provider.SEP_ID] 
        inputs += list(map(lambda s: self.vocab.get(s, self.vocab_provider.UNK_ID) + vocab_size * d2, w2.lower()))
        inputs += [self.vocab_provider.PAD_ID]*(maxlen - len(inputs))
        
        tags = get_overlaps(w1, w2, self.p1, self.p2)
        tags += [[0]]*(maxlen - len(tags))

        y = np.array(inputs)
        y = np.expand_dims(y, axis=0)
        v = np.array(tags)
        v = np.expand_dims(v, axis=0)

        if flip:
            inputs.reverse()
            tags.reverse()

        x = np.array(inputs)
        x = np.expand_dims(x, axis=0)
        u = np.array(tags)
        u = np.expand_dims(u, axis=0)

        return x, y, u, v
    
    def iterate(self, iterations, batch_size, set_name=None):
        data = self.data_set[set_name][0] if set_name else self.full
        for i in range(iterations):
            indices = np.random.randint(len(data), size=batch_size)
            x = np.array([data[index][0] for index in indices])
            y = np.copy(x)
            x = np.flip(x, 1)
            u = np.array([data[index][1] for index in indices]).reshape(-1, self.maxlen_source, 1)
            v = np.copy(u)
            u = np.flip(u, 1)
            z = np.array([data[index][2] for index in indices])
            yield x,y,z,u,v
        

    def iterate_seq(self, batch_size, set_name=None):
        data = self.data_set[set_name][0] if set_name else self.full
        for i in range(0, len(data)//batch_size*batch_size, batch_size):
            x = np.array(list(zip(*data))[0][i:i+batch_size])
            y = np.copy(x)
            x = np.flip(x, 1)
            u = np.array(list(zip(*data))[1][i:i+batch_size]).reshape(-1, self.maxlen_source, 1)
            v = np.copy(u)
            u = np.flip(u, 1)
            z = np.array(list(zip(*data))[2][i:i+batch_size])
            yield x,y,z,u,v

    def vocab_size(self):
        return len(self.vocab)

    def seq_sizes(self):
        return self.maxlen_source, self.maxlen_target

    def dataset_size(self, name):
        if name is None:
            return len(self.full)
        return len(self.data_set[name][0])

    def get_texts(self, partition_name=None):
        if len(self.data_set) > 0 and partition_name:
            data, offset = self.data_set[partition_name]
            src = self.sources[offset:offset + len(data)]
            tgt = self.targets[offset:offset + len(data)]
        else:
            src = self.sources
            tgt = self.targets

        return src, tgt


    