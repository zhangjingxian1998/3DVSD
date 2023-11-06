import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser = parser.parse_args()
    parser = vars(parser)
    with open('vsd_3d/config/Config.yaml', 'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
    parser.update(config)
    parser = argparse.Namespace(**parser)
    # args = Config(**kwargs)
    return parser