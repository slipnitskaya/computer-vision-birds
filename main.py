import os
import tqdm
import torch
import pickle
import argparse
import numpy as np
import torchvision as tv
import torch.utils.data as td
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


class Attention(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Attention, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                     dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = torch.nn.Tanh()

    def forward(self, inputs):
        x, output_size = inputs
        x = F.adaptive_max_pool2d(x, output_size=output_size)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x) + 1.0

        return x


class ResNet18(torch.nn.Module):

    def __init__(self,
                 num_classes: int = 200,
                 pretrained: bool = False,
                 use_attention: bool = False,
                 grad_center: bool = False):

        super(ResNet18, self).__init__()

        self.use_attention = use_attention
        self.pretrained = pretrained
        self.grad_center = grad_center

        self.net = tv.models.resnet18(pretrained=pretrained)
        self.net.fc = torch.nn.Linear(
            in_features=self.net.fc.in_features,
            out_features=num_classes,
            bias=True
        )

        if use_attention:
            self.att1 = Attention(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
            self.att2 = Attention(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
            self.att3 = Attention(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
            self.att4 = Attention(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)

            if pretrained:
                self.att1.bn.weight.data.zero_()
                self.att1.bn.bias.data.zero_()
                self.att2.bn.weight.data.zero_()
                self.att2.bn.bias.data.zero_()
                self.att3.bn.weight.data.zero_()
                self.att3.bn.bias.data.zero_()
                self.att4.bn.weight.data.zero_()
                self.att4.bn.bias.data.zero_()

        if grad_center:
            for p in self.parameters():
                p.register_hook(lambda grad: grad - grad.mean())

    def forward(self, x):
        if self.use_attention:
            return self._forward_attn(x)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x

    def _forward_attn(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x_a = x.clone()
        x = self.net.layer1(x)
        x = x * self.att1((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer2(x)
        x = x * self.att2((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer3(x)
        x = x * self.att3((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer4(x)
        x = x * self.att4((x_a, x.shape[-2:]))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class DatasetBirds(tv.datasets.ImageFolder):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True):

        img_root = os.path.join(root, 'images')

        super(DatasetBirds, self).__init__(
            root=img_root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.train = train

        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        imgs_to_use = list()
        for path_to_img, lb in self.imgs:
            fn_to_exclude = None

            for fn in filenames_to_use:
                if path_to_img.endswith(fn):
                    imgs_to_use.append((path_to_img, lb))
                    fn_to_exclude = fn
                    break

            if fn_to_exclude is not None:
                filenames_to_use -= {fn_to_exclude}

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

    def __getitem__(self, index):
        return super(DatasetBirds, self).__getitem__(index)

    def __len__(self):
        return len(self.targets)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--in-dir', default='data/CUB_200_2011')
    parser.add_argument('-o', '--out-dir', default='results')
    parser.add_argument('-t', '--pretrained', default=False)
    parser.add_argument('-a', '--use-attention', default=False)
    parser.add_argument('-g', '--grad-center', default=False)
    parser.add_argument('-b', '--batch-size', default=128)
    parser.add_argument('-e', '--num-epochs', default=50)
    parser.add_argument('-w', '--num-workers', default=8)
    parser.add_argument('-s', '--save-results', default=False)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare dataset
    transforms_train = tv.transforms.Compose([
        tv.transforms.Resize((256, 256)),
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_eval = tv.transforms.Compose([
        tv.transforms.Resize((256, 256)),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds_train = DatasetBirds(args.in_dir, transform=transforms_train, train=True)
    ds_val = DatasetBirds(args.in_dir, transform=transforms_eval, train=True)
    ds_test = DatasetBirds(args.in_dir, transform=transforms_eval, train=False)

    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

    train_loader = td.DataLoader(
        dataset=ds_train, batch_size=args.batch_size, sampler=td.SubsetRandomSampler(idx_train), num_workers=args.num_workers
    )
    val_loader = td.DataLoader(
        dataset=ds_val, batch_size=args.batch_size, sampler=td.SubsetRandomSampler(idx_val), num_workers=args.num_workers
    )
    test_loader = td.DataLoader(
        dataset=ds_test, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # instantiate the model
    model = ResNet18(
        num_classes=len(ds_train.classes),
        pretrained=args.pretrained,
        use_attention=args.use_attention,
        grad_center=args.grad_center
    ).to(device)

    # instantiate optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_loss_avg = list()
    train_acc_avg = list()
    val_loss_avg = list()
    val_acc_avg = list()
    log_interval = 20
    for _ in tqdm.trange(args.num_epochs):
        train_loss = list()
        train_acc = list()
        train_loss_per_batch = 0.0
        bar = tqdm.tqdm(total=len(train_loader), leave=False)
        # train the model
        model.train()
        for i, batch in enumerate(train_loader):
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss_per_batch += loss.item()
            acc = accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

            if (i + 1) % log_interval == 0:
                bar.set_description('Train loss={:.3f}'.format(loss.item()), True)
                bar.update()

            train_loss.append(loss.item())
            train_acc.append(acc)

        train_loss_avg.append(np.mean(train_loss))
        train_acc_avg.append(np.mean(train_acc))

        # validate the model
        model.eval()
        with torch.no_grad():
            val_loss = list()
            val_acc = list()
            bar = tqdm.tqdm(total=len(val_loader), leave=False)
            for i, batch in enumerate(val_loader):
                x, y = batch

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                loss = F.cross_entropy(y_pred, y)
                acc = accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

                if (i + 1) % log_interval == 0:
                    bar.set_description('Valid loss={:.3f}'.format(loss.item()), True)
                    bar.update()

                val_loss.append(loss.item())
                val_acc.append(acc)

            val_loss_avg.append(np.mean(val_loss))
            val_acc_avg.append(np.mean(val_acc))

        scheduler.step()

    # test the model
    true = list()
    pred = list()
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(test_loader, leave=False)):
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            true.extend([val.item() for val in y])
            pred.extend([val.item() for val in y_pred.argmax(dim=-1)])

    print(f'Test accuracy: {accuracy_score(true, pred)}')

    # save results
    if args.save_results:
        exp_id = ''.join([
            'pretrained' if model.pretrained else '',
            'Attention' if model.use_attention else '',
            'Grad' if model.grad_center else ''
        ])
        with open(f'{args.out_dir}/performance_{exp_id}.pkl', 'wb') as f:
            metrics = {
                'train_loss': train_loss_avg, 'val_loss': val_loss_avg, 'train_acc': train_acc_avg, 'val_acc': val_acc_avg
            }
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{args.out_dir}/predictions_{exp_id}.pkl', 'wb') as f:
            predictions = {'y_true': true, 'y_pred': pred, 'labels': ds_train.classes}
            pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
