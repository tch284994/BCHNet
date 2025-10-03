r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL

from data.fss import DatasetFSS
from data.deepglobe import DatasetDeepglobe
from data.isic import DatasetISIC
from data.lung import DatasetLung

from data.verse2D_axial import DatasetVerse2D_Axial
from data.verse2D_sagittal import DatasetVerse2D_Sagittal


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {

            # source domain
            'pascal': DatasetPASCAL,

            # target domain
            'fss': DatasetFSS,
            'deepglobe': DatasetDeepglobe,
            'isic': DatasetISIC,
            'lung': DatasetLung,

            # spinal segmentation
            'verse2D_axial': DatasetVerse2D_Axial,
            'verse2D_sagittal': DatasetVerse2D_Sagittal,
        }
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=bsz, num_workers=nworker, drop_last=True, sampler=sampler, pin_memory=True)
        return dataloader, sampler
