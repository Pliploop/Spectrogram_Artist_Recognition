import yaml
import os
import sys
import argparse
from Preprocessing_Pipeline import Splitting_Pipeline,SegmentPipeline

parser = argparse.ArgumentParser()
parser.add_argument('-p','--preprocess', required=False)
parser.add_argument('-t','--train', required=False)
parser.add_argument('-s','--score', required=False)
parser.add_argument('-sp','--split', required=False)
parser.add_argument('-bd','--builddataset', required=False)

PreProc = False
Train = False
Score = False

args = parser.parse_args()

if args.preprocess:
    PreProc = True
    
if args.train:
    Train = True
    
if args.score:
    Score = True


if __name__ == '__main__':
    with open("config/config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    if PreProc:
        Splitting = Splitting_Pipeline(config['PreProc'])
        Splitting.run_pipeline(args.split,args.builddataset)
        Segmenting = SegmentPipeline(config['PreProc'])
    pass
