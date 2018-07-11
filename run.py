from caver import Trainer, Caver, Ensemble


if __name__ == '__main__':
    # train model
    t = Trainer('CNN', '/data/train.fasttext')

    t.train()

    # classify
    cnn = Caver('CNN', 'checkpoint/CNN_10.pth')
    print(cnn.predict('NBA'))
    print(cnn.get_top_label('nba'))

    # train SWEN
    t = Trainer('SWEN', '/data/train.fasttext')
    t.train()

    # classify
    swen = Caver('SWEN', 'SWEN_10.pth')
    print(swen.predict('NBA'))
    print(swen.get_top_label('nba'))

    # train LSTM
    t = Trainer('LSTM', '/data/train.fasttext')
    t.train()

    # classify
    lstm = Caver('LSTM', 'LSTM_10.pth')
    print(lstm.predict('nba'))
    print(lstm.get_top_label('nba'))

    # train HAN
    t = Trainer('HAN', '/data/train.fasttext')
    t.train()

    # classify
    han = Caver('HAN', 'HAN_10.pth')
    print(han.predict('nba'))
    print(han.get_top_label('nba'))

    # ensemble
    model = Ensemble([cnn, swen, lstm, han])
    print(model.get_top_label('nba'))

