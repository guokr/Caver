import caver.model.cavermodel as cavermodel
import caver.ensemble_m.ensemble as ensemble
from decimal import Decimal
from decimal import ROUND_HALF_UP


def load_models():
    model_lstm = cavermodel.CaverModel("/data_hdd/caver_models/checkpoints_char_lstm", device="cpu")
    model_cnn = cavermodel.CaverModel("/data_hdd/caver_models/checkpoints_char_cnn", device="cpu")
    models = [model_lstm, model_cnn]
    return models


def myRound(num):
    return Decimal(num).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

# def get_preds_by_mean(models, model_ratio, models_preds):
#     ensem_model = ensemble.EnsembleModel(models, model_ratio)
#     ensem_model_preds = ensem_model.mean(models_preds)
#     return ensem_model_preds



