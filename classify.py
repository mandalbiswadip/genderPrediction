from __future__ import unicode_literals
from __future__ import print_function
import argparse

from scripts import train

parser = argparse.ArgumentParser(description='Classify Indian names')

parser.add_argument('name', type=str, help='Name to be classified')

args = parser.parse_args()

print("Gender for {}: {}".format(args.name, train.classify(args.name)))