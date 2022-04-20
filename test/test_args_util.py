import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import argparse
from utils.args_util import print_args



parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir100')
parser.add_argument('--threshold', type=float, default='0.5')
parser.add_argument('--lr', type=float, default='1e-3')
parser.add_argument('--data_loader_type', type=int, default='0')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help="choose from ['cosine', 'custom', 'constant']")
parser.add_argument('--loss_type', type=str, default='ce', help="choose from ['ce', 'fl', 'refl']")


if __name__ == "__main__":
    args = parser.parse_args()
    print_args(args)