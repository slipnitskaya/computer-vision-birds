import tqdm
import torch
import numpy as np
import torchvision as tv
import torch.utils.data as td
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Attention(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Attention, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                                     padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        x, output_size = inputs
        x = F.adaptive_max_pool2d(x, output_size=output_size)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ResNet18(torch.nn.Module):

    def __init__(self, num_classes=200, pretrained=False):
        super(ResNet18, self).__init__()

        net = tv.models.resnet18(pretrained=pretrained)

        net.fc = torch.nn.Linear(
            in_features=net.fc.in_features,
            out_features=num_classes,
            bias=True
        )

        self.net = net

    def forward(self, x):
        return self.net(x)


class ResNet18Attention(ResNet18):

    def __init__(self, num_classes=200, pretrained=True, use_attention=True):
        super(ResNet18Attention, self).__init__(num_classes=num_classes, pretrained=pretrained)

        self.use_attention = use_attention

        if use_attention:
            self.att1 = Attention(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
            self.att2 = Attention(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
            self.att3 = Attention(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
            self.att4 = Attention(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)

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


def main():
    in_dir = 'data/CUB_200_2011/images'
    out_dir = 'results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define setups
    batch_size = 128
    num_epochs = 50
    num_workers = 8
    random_seed = 42
    log_step = 20
    learning_rate = 1e-3
    pretrained = False

    # prepare dataset
    transforms = tv.transforms.Compose([tv.transforms.Resize(size=(224, 224)), tv.transforms.ToTensor()])
    ds = tv.datasets.ImageFolder(in_dir, transform=transforms)

    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    idx_train, idx_temp = next(splits.split(np.zeros(len(ds)), ds.targets))
    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)
    idx_val, idx_test = next(splits.split(np.zeros(len(idx_temp)), [ds.targets[idx] for idx in idx_temp]))
    idx_val = idx_temp[idx_val]
    idx_test = idx_temp[idx_test]

    train_loader = td.DataLoader(
        dataset=ds, batch_size=batch_size, sampler=td.SubsetRandomSampler(idx_train), num_workers=num_workers
    )
    val_loader = td.DataLoader(
        dataset=ds, batch_size=batch_size, sampler=td.SubsetRandomSampler(idx_val), num_workers=num_workers
    )
    test_loader = td.DataLoader(
        dataset=ds, batch_size=batch_size, sampler=td.SubsetRandomSampler(idx_test), num_workers=num_workers
    )

    # setup the model
    model = ResNet18(num_classes=len(ds.classes), pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train and validate the model
    train_loss_avg = list()
    val_loss_avg = list()
    train_acc_avg = list()
    val_acc_avg = list()
    for _ in tqdm.trange(num_epochs):
        train_loss = list()
        train_acc = list()
        train_loss_per_batch = 0.0
        bar = tqdm.tqdm(total=len(train_loader), leave=False)
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

            if (i + 1) % log_step == 0:
                bar.set_description('Train loss={:.3f}'.format(loss.item()), True)
                bar.update()

            train_loss.append(loss.item())
            train_acc.append(acc)

        train_loss_avg.append(np.mean(train_loss))
        train_acc_avg.append(np.mean(train_acc))

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

                if (i + 1) % log_step == 0:
                    bar.set_description('Valid loss={:.3f}'.format(loss.item()), True)
                    bar.update()

                val_loss.append(loss.item())
                val_acc.append(acc)

            val_loss_avg.append(np.mean(val_loss))
            val_acc_avg.append(np.mean(val_acc))

    # test the model
    gt = list()
    pred = list()
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(test_loader, leave=False)):
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            gt.extend([val.item() for val in y])
            pred.extend([val.item() for val in y_pred.argmax(dim=-1)])

        test_acc = accuracy_score(gt, pred)

    # save summary results
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    plt.suptitle('Model performance. Test accuracy: {:.2f}.'.format(test_acc))
    ax[0].plot(train_loss_avg, label='training')
    ax[0].plot(val_loss_avg, label='validation')
    ax[1].plot(train_acc_avg)
    ax[1].plot(val_acc_avg)
    ax[0].set_ylabel('loss')
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epochs')
    ax[0].legend()
    plt.savefig(f'{out_dir}/performance.png', bbox_inches='tight')
    plt.close(fig)

    cm = confusion_matrix(y_true=gt, y_pred=pred, labels=range(len(set(ds.classes))))
    cm_nodiag = cm * ~np.eye(*cm.shape, dtype=bool)
    confused_ids = cm_nodiag.sum(-1) == cm_nodiag.sum(-1).max()
    cm_confused = cm[confused_ids][:, confused_ids] + cm[confused_ids, confused_ids]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    hm = ConfusionMatrixDisplay(cm_confused, [l.split('.')[-1] for l in np.array(ds.classes)[confused_ids]])
    hm = hm.plot(include_values=False, ax=ax, cmap='Blues', xticks_rotation=90)
    num_confused = np.sum(confused_ids).item()
    if num_confused > 25:
        plt.xticks(range(0, num_confused), [])
        plt.yticks(range(0, num_confused), [])
        hm.im_.set_clim(0, 1)
    plt.title(f'Most confused classes: {num_confused}.', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Confused', fontsize=12)
    plt.savefig(f'{out_dir}/confmatrix{"_norm" if num_confused > 25 else ""}.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
