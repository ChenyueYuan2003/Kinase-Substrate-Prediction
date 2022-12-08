import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--data_repeat", default=10, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)

    parser.add_argument("--data_path", default='./data/samples/', type=str)
    parser.add_argument("--model_path", default='./model/', type=str)

    return parser.parse_args()
