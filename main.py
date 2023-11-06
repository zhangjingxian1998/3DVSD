from vsd_3d.datasets import VSD_3d_dataset
import torch
from vsd_3d.utility import parse_args
from vsd_3d.models import Model
from tqdm import tqdm
# from rich.progress import track
def main(cfg):
    vsd_data = VSD_3d_dataset(cfg)
    vsd_data_dataloader = torch.utils.data.DataLoader(vsd_data,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers)
    model = Model()
    for item in tqdm(vsd_data_dataloader):
        model.train()
        model.train_epoch(item)
    pass

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)