import argparse

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Total 3D Understanding.')
    parser.add_argument('--config', type=str, default='configs/total3d.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='demo', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
    return parser.parse_args()