
import argparse
import os

from parse_data_from_sportscap import create_polevault_or_highjump_dataset, create_diving_dataset, create_gymnastics_dataset
from parse_data_from_ski import create_ski_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str, default='ski')
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()


if __name__ == '__main__':
    out_path = './external/DAPA_release/data/dataset_extras'

    if args.action.startswith('ski'):
        num_samples = int(args.action.strip('ski'))
        create_ski_dataset(out_path, action=args.action, num_train_samples=num_samples, split='train', debug=args.debug)
        create_ski_dataset(out_path, action=args.action, num_train_samples=num_samples, split='test', debug=args.debug)
        
    elif args.action in ['polevault', 'highjump']:
        create_polevault_or_highjump_dataset(out_path, action=args.action, split='train')
        create_polevault_or_highjump_dataset(out_path, action=args.action, split='test')
    
    elif args.action == 'diving':
        create_diving_dataset(out_path, split='train')
        create_diving_dataset(out_path, split='test')
    
    elif args.action in ['balancebeam', 'vault', 'unevenbars']:
        create_gymnastics_dataset(out_path, action=args.action, split='train')
        create_gymnastics_dataset(out_path, action=args.action, split='test')

    else:
        raise ValueError(f'Action {args.action} not supported')
