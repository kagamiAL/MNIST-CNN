import json
import numpy.typing as nt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as torch_data
from net import Net
from mnist_dataset import MNIST_Dataset

# Print iterations progress


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_np_csv(path: str) -> nt.ArrayLike:
    return np.loadtxt(path, delimiter=",", dtype=np.int64)


def normalize(data: torch.Tensor):
    return (data - torch.mean(data)) / torch.std(data)


def get_dataloader(
    dataset: torch_data.Dataset, batch_size: int
) -> torch_data.DataLoader:
    return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_data(path, transform=None) -> torch_data.Dataset:
    digit_np = get_np_csv(path)
    digit_t = torch.from_numpy(digit_np)
    one_hot = torch.zeros(digit_t.shape[0], 10)
    one_hot.scatter_(1, digit_t[:, 0].unsqueeze(1), 1.0)
    return MNIST_Dataset(
        normalize(digit_t[:, 1:].float()).view(-1, 28, 28).unsqueeze(1).to("cuda"),
        one_hot.to("cuda"),
        transform,
    )


def get_accuracy(model, data):
    correct = 0
    total = 0
    with torch.no_grad():
        for input, labels in data:
            outputs = model(input)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == torch.argmax(labels, dim=1)).sum())
    return correct / total


def training_loop(
    n_epochs: int,
    optimizer: optim.SGD,
    model: Net,
    train_loader: torch_data.DataLoader,
    val_loader: torch_data.DataLoader,
):
    loss_fn = nn.CrossEntropyLoss()
    pre = "Progress per 10 Epochs"
    for epoch in range(1, n_epochs + 1):
        avg_loss_train = 0.0
        total = 0
        printProgressBar((epoch - 1) % 10, 10, prefix=pre)
        for input, labels in train_loader:
            loss = loss_fn(model(input), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss_train += loss.item()
            total += 1
        printProgressBar((epoch - 1) % 10 + 1, 10, prefix=pre)
        if epoch % 10 == 0:
            print(f"""\nEpoch: {epoch}
            Avg train loss: {avg_loss_train/total}
            Val accuracy: {get_accuracy(model, val_loader)}
            Train accuracy: {get_accuracy(model, train_loader)}
            """)


def get_params():
    with open("./parameters.json") as f:
        return json.load(f)


if __name__ == "__main__":
    params = get_params()
    train_dataset = load_data("./MNIST_CSV/mnist_train.csv")
    val_dataset = load_data("./MNIST_CSV/mnist_test.csv")
    train_loader = get_dataloader(train_dataset, params["batch_size"])
    val_loader = get_dataloader(val_dataset, params["batch_size"])
    model = Net().to("cuda")
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
    training_loop(params["epochs"], optimizer, model, train_loader, val_loader)
    # TODO: Fix saving with lowest val loss
    torch.save(model.state_dict(), params["name"])
