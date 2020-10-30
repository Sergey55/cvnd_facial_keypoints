from model import Net
from datamodule import FacialKeypointsDatamodule

from pytorch_lightning import Trainer
from argparse import ArgumentParser

def main(args):
    dm = FacialKeypointsDatamodule(
        root_dir='./data/',
        train_csv_file='./data/training_frames_keypoints.csv',
        test_csv_file='./data/test_frames_keypoints.csv',
        num_workers=8
    )
    model = Net()

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)

    main(parser.parse_args())