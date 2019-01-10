import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import argparse
from models.module import DataParallelModel
from data import Flickr, Icdar
from torch.utils.data.dataloader import DataLoader as Dataloader
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from models.module import Model
from util import load_ckpt, save_ckpt, visualization
from tensorboardX import SummaryWriter
from os.path import join


def test(model, dataloader, dir):
    model.eval()
    for index, data in enumerate(dataloader):
        with torch.no_grad():
            _, out_dict = model(data)
            grid = visualization(out_dict)
            save_image(grid, join(dir, '{}.jpg'.format(index)))


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--snapshot_dir', type=str, default='./snapshot')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--resume', type=int)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--vis_interval', type=int, default=20)
args = parser.parse_args()

if not os.path.exists(args.snapshot_dir):
    os.makedirs('{:s}/images'.format(args.snapshot_dir))
    os.makedirs('{:s}/ckpt'.format(args.snapshot_dir))

# model construction
model = Model()
model = DataParallelModel(model)
model = model.cuda()

train_set = Flickr('./q')
test_set = Flickr('./q')
# test_set = Icdar('./icdartest')
train_loader = Dataloader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True,
                          drop_last=False)
test_loader = Dataloader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

if isinstance(model, torch.nn.DataParallel):
    params = model.module.parameters()
else:
    params = model.parameters()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.weight_decay)

# model load
if args.resume:
    if isinstance(model, torch.nn.DataParallel):
        start_epoch = load_ckpt(args.resume, [('model', model.module)], [('optimizer', optimizer)])
    else:
        start_epoch = load_ckpt(args.resume, [('model', model)], [('optimizer', optimizer)])

else:
    start_epoch = 0

writer = SummaryWriter()

for i in tqdm(range(start_epoch + 1, args.max_epoch + 1)):
    model.train()
    epoch_loss = []
    for in_dict in train_loader:
        for _, v in in_dict.items():
            v = v.cuda()
        loss_dict, _ = model(in_dict)
        loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(loss_name, float(sum(loss_value)), i)
            loss += loss_value
            epoch_loss.append(float(sum(loss)))
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
    print('epoch loss : {}'.format(sum(epoch_loss) / len(epoch_loss)))

    if i % args.save_interval == 0 or i == args.max_epoch:
        if isinstance(model, torch.nn.DataParallel):
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.snapshot_dir, i), [('model', model.module)],
                      [('optimizer', optimizer)], i)
        else:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.snapshot_dir, i), [('model', model)], [('optimizer', optimizer)],
                      i)
    if i % args.vis_interval == 0 or i == args.max_epoch:
        if not os.path.exists('./snapshot/images/{:d}'.format(i)):
            os.makedirs('./snapshot/images/{:d}'.format(i))
        test_dir = './snapshot/images/{}'.format(i)
        test(model, test_loader, test_dir)
