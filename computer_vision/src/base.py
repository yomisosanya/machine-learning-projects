import argparse

def add_hyper_params(parser: argparse.ArgumentParser):
    parser.add_argument('--input-size')
    parser.add_argument('--output-size')