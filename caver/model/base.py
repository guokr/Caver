import os
import torch

class BaseModule(torch.nn.Module):
    """
    Base module for text classification.

    Inherit this if you want to implement your own model.
    """
    def __init__(self):
        super().__init__()

    def load(self, path):
        """ load model from file """
        assert os.path.isfile(path)
        self.load_state_dict(torch.load(path,
                    map_location=lambda storage, loc: storage))

    def save(self, path):
        """ save model to file """
        folder, _ = os.path.split(path)
        if not os.path.isdir(folder):
            os.mkdir(folder)
            print('Folder: {} is created.'.format(folder))

        torch.save(self.state_dict(), path)
        print('[+] Model saved.')

    def get_args(self):
        return vars(self)


    def update_args(self, args):
        for arg, value in args.items():
            vars(self)[arg] = value

