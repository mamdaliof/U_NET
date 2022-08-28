from Model import  test
import argparse
from Train import train

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='classification')

    parser.add_argument("--epochs", dest='epochs', default=5)

    parser.add_argument("--batch_size", dest='batch_size', default=32)

    parser.add_argument("--learning_rate", dest='main_lr', default=0.01)

    parser.add_argument("--train_path", dest='train_path',
                        default="../cityscapes_data/train")

    parser.add_argument("--val_path", dest='val_path',
                        default="../cityscapes_data/val")

    parser.add_argument("--in_channels", dest='in_channels', default=3)

    parser.add_argument("--out_channels", dest='out_channels', default=3)

    return parser.parse_args()


def main():
    args =arg_parse()
    my_train = train(args, True)
    my_train.execute()

def t():
    test()
if __name__ == '__main__':
    t()
    # main()
