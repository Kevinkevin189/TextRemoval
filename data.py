from torch.utils.data import Dataset as dataset
from torchvision.transforms import ToTensor, RandomResizedCrop
from torchvision.transforms.functional import resized_crop
from os.path import join
from os import listdir
from PIL import Image


class Flickr(dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.origin_list = listdir(join(data_path, 'origin'))
        self.transform = ToTensor()

    def __len__(self):
        return len(self.origin_list)

    def process_img(self, origin, mask):
        i, j, h, w = RandomResizedCrop.get_params(origin, scale=(0.5, 2.0), ratio=(3.0 / 4, 4.0 / 3))
        origin = resized_crop(origin, i, j, h, w, size=(512, 512), interpolation=Image.NEAREST)
        mask = resized_crop(mask, i, j, h, w, size=(512, 512), interpolation=Image.NEAREST)
        origin = self.transform(origin)
        mask = self.transform(mask)
        mask = mask.squeeze(0).long()
        return origin, mask

    def __getitem__(self, index):
        assert index < len(self), 'index out of range error'
        file = self.origin_list[index]
        origin = Image.open(join(self.data_path, 'origin', file)).convert('RGB')
        mask = Image.open(join(self.data_path, 'mask', file.strip('jpg') + 'bmp')).convert('1')
        data_dict = {}
        data_dict['origin'], data_dict['mask'] = self.process_img(origin, mask)
        return data_dict


class Icdar(dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.origin_list = listdir(join(self.data_path, 'origin'))
        self.transform = ToTensor()

    def __len__(self):
        return len(self.origin_list)

    def process_img(self, origin, mask):
        i, j, h, w = RandomResizedCrop.get_params(origin, scale=(0.5, 2.0), ratio=(3.0 / 4, 4.0 / 3))
        origin = resized_crop(origin, i, j, h, w, size=(640, 480), interpolation=Image.NEAREST)
        mask = resized_crop(mask, i, j, h, w, size=(640, 480), interpolation=Image.NEAREST)
        origin = self.transform(origin)
        mask = self.transform(mask)
        mask = mask.squeeze(0).long()
        return origin, mask

    def __getitem__(self, index):
        assert index < len(self), 'index out of range error'
        file = self.origin_list[index]
        origin = Image.open(join(self.data_path, 'origin', file)).convert('RGB')
        mask = Image.open(join(self.data_path, 'mask', file.strip('jpg') + 'png')).convert('L')
        data_dict = {}
        data_dict['origin'], data_dict['mask'] = self.process_img(origin, mask)
        return data_dict


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader as dataloader

    # dl=dataloader(dataset=Flickr('./data'), batch_size=2, shuffle=True)
    dl = dataloader(dataset=Icdar('./icdartest'), batch_size=2, shuffle=True)
    for values in dl:
        print(values.shape)
