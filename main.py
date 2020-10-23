import tqdm
import torch
import numpy as np
import torchvision as tv
import torch.utils.data as td
import sklearn.metrics as skm
import torch.nn.functional as F

from sklearn.model_selection import StratifiedShuffleSplit


def main():
    in_dir = 'data/CUB_200_2011/images'
    batch_size = 128
    num_epochs = 100
    num_workers = 6
    random_seed = 42
    learning_rate = 1e-2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # define the model
    model = tv.models.resnet18(num_classes=len(ds.classes)).to(device)

    # model training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in tqdm.trange(num_epochs):
        model.train()

        bar = tqdm.tqdm(total=len(train_loader), leave=False)
        for batch in train_loader:
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)

            loss.backward()
            optimizer.step()

            bar.set_description('Loss={:.3f}'.format(loss.item()), False)
            bar.update()

        # validation
        model.eval()
        with torch.no_grad():
            bar = tqdm.tqdm(total=len(val_loader), leave=False)
            for batch in val_loader:
                x, y = batch

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = F.cross_entropy(y_pred, y)

                bar.set_description('Loss={:.3f}'.format(loss.item()), False)
                bar.update()

    # testing
    gt = list()
    pred = list()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, leave=False):
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            gt.extend([val.item() for val in y])
            pred.extend([val.item() for val in y_pred.argmax(dim=-1)])

    print('Test accuracy: {:.2f}'.format(skm.accuracy_score(gt, pred)))


if __name__ == '__main__':
    main()
