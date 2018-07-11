import os
import json
from collections import Counter

from .config import Config
from .utils import update_config

class TextData:
    def __init__(self, path='', **kwargs):
        """
        data format: fastText format, each line contains a list of labels start
        by `__label__` prefix and words split by space. (Chinese should be
        segmented before being used.)
        """
        self.path = path
        self.config = update_config(Config(), **kwargs)

        if os.path.isfile(self.config.word2index) and os.path.isfile(self.config.label2index):
            self.load_index()
        else:
            words, labels = self.extract()
            self.build_index(words, labels)


    def load_index(self):
        """
        Load index information from JSON file.
        """
        print('Loading index from local file...')

        with open(self.config.label2index, 'r', encoding='utf-8') as f:
            self.label2index = json.load(f)

        with open(self.config.word2index, 'r', encoding='utf-8') as f:
            self.word2index = json.load(f)

        print('Load {} words and {} labels.'.format(
            len(self.word2index), len(self.label2index)
        ))

    def extract(self):
        """
        Extract word-freq and label-freq from data file.
        """
        assert os.path.isfile(self.path)
        print('Generating index from data file...')
        words, labels = Counter(), Counter()

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().lower().split(' ')
                label = [x for x in items if x.startswith('__label__')]
                word = [x for x in items if not x.startswith('__label__')]
                if label and word:
                    labels.update(label)
                    words.update(word)

        print('Extract {} words and {} labels.'.format(
            len(words), len(labels)
        ))
        return words, labels

    def build_index(self, words, labels):
        self.word2index = {}
        for i, (word, freq) in enumerate(words.most_common()):
            if freq < self.config.min_word_count:
                break
            self.word2index[word] = i

        self.label2index = {}
        for i, (label, freq) in enumerate(labels.most_common()):
            if freq < self.config.min_label_count:
                break
            self.label2index[label] = i

        if not os.path.isdir(self.config.index_path):
            os.mkdir(self.config.index_path)
            print('Index path {} is created.'.format(self.config.index_path))

        with open(os.path.join(self.config.index_path, 'word2index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.word2index, f)

        with open(os.path.join(self.config.index_path, 'label2index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.label2index, f)

    def prepare(self):
        """
        Generate data replaced by index from data file.
        """
        x, y = [], []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().lower().split(' ')
                label = [self.label2index.get(a) for a in items if a.startswith('__label__')]
                word = [self.word2index.get(a) for a in items if not a.startswith('__label__')]
                if word and label:
                    x.append(word)
                    y.append(label)

        return x, y


class Segment:
    """
    :param model: model type, ['jieba', 'pyltp']
    :type model: str
    :param userdict: user dict file, used for initializing segment model
    :type userdict: str
    :param model_path: segment model path (if you use `pyltp`)
    :type model_path: str
    """
    def __init__(self, model='jieba', userdict=None, model_path=None):
        self.model = model
        if model == 'jieba':
            import jieba
            self.seg = jieba
            if userdict and os.path.isfile(userdict):
                self.seg.load_userdict(userdict)
            self.seg.initialize()
        elif model == 'pyltp':
            import pyltp
            self.seg = pyltp.Segmentor()
            assert os.path.isfile(model_path)
            if userdict:
                self.seg.load_with_lexicon(model_path, userdict)
            else:
                self.seg.load(model_path)
        else:
            print('Use `Plane.segment` to cut sentence.')
            import plane
            self.seg = plane.segment

    def cut(self, text):
        """
        Cut sentence into words list.
        """
        if self.model == 'jieba':
            return self.seg.lcut(text)
        elif self.model == 'pyltp':
            return list(self.seg.segment(text))
        else:
            return self.seg(text)
