import tqdm
import torch
import numpy as np
import torchvision as tv
import torch.utils.data as td
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    in_dir = 'data/CUB_200_2011/images'
    out_dir = 'results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define setups
    batch_size = 128
    num_epochs = 100
    num_workers = 8
    random_seed = 42
    learning_rate = 1e-2
    log_step = 20

    # prepare dataset
    transforms = tv.transforms.Compose([tv.transforms.Resize(size=(224, 224)), tv.transforms.ToTensor()])
    ds = tv.datasets.ImageFolder(in_dir, transform=transforms)

    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    idx_train, idx_test = next(splits.split(np.zeros(len(ds)), ds.targets))
    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)
    idx_val, idx_test = next(splits.split(np.zeros(len(idx_test)), [ds.targets[idx] for idx in idx_test]))

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
    model = tv.models.resnet18(num_classes=len(ds.classes)).to(device)
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
    plt.suptitle('Model performance. Test accuracy: {:.2f}'.format(test_acc))
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


if __name__ == '__main__':
    main()
