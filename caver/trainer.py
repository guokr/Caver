import tqdm
from caver.evaluator import Evaluator
import torch
import os


class Trainer:
    def __init__(self, model, optimizer, loss):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss

    def train_step(self, train_data, epoch, config):
        evaluator = Evaluator(self.criterion)
        self.model.train()
        tqdm_progress = tqdm.tqdm(train_data, desc="| Training epoch {}/{}".format(epoch, config.epoch))
        for x, y in tqdm_progress:
            self.optimizer.zero_grad()
            preds = self.model(x)
            ev = evaluator.evaluate(preds, y)
            self.optimizer.step()

            tqdm_progress.set_postfix({"Loss": "{:.4f}".format(ev[0]),
                                       "Recall": "{:.4f}".format(ev[1]),
                                       "Precsion": "{:.4f}".format(ev[2]),
                                       "F_Score": "{:.4f}".format(ev[3])})

    def valid_step(self, model_args, valid_data, epoch, config):
        evaluator = Evaluator(self.criterion)
        self.model.eval()
        tqdm_progress = tqdm.tqdm(valid_data, desc="| Validating epoch {}/{}".format(epoch, config.epoch))
        for x, y in tqdm_progress:
            if x.size(1) < 4:
                print("ok minibatch skiped")
                continue

            preds = self.model(x)
            ev = evaluator.evaluate(preds, y, mode="eval")
            tqdm_progress.set_postfix({"Loss": "{:.4f}".format(ev[0]),
                                       "Recall": "{:.4f}".format(ev[1]),
                                       "Precsion": "{:.4f}".format(ev[2]),
                                       "F_Score": "{:.4f}".format(ev[3])
                                       })
        torch.save({"model_type": config.model,
                    "model_args": model_args,
                    "model_state_dict": self.model.state_dict()},
                   os.path.join(config.checkpoint_dir, "checkpoint_{}.pt".format(epoch)))

        return ev[0]



