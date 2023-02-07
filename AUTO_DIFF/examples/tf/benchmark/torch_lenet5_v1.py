import torch
import torch.utils.data as Data
from torch import nn
from tensorflow.keras.datasets import mnist
import numpy as np
import time

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='valid'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='valid'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=1600, out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=10)
)
model = model.cuda()

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x[:, None, ...].astype(np.float32) / 255.
test_x = test_x[:, None, ...].astype(np.float32) / 255.
train_x = np.pad(train_x, [[0, 0], [0, 0], [2, 2], [2, 2]])
test_x = np.pad(test_x, [[0, 0], [0, 0], [2, 2], [2, 2]])
train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)
train_x, train_y, test_x, test_y = map(torch.from_numpy, [train_x, train_y, test_x, test_y])
train_iter = Data.DataLoader(Data.TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
test_iter = Data.DataLoader(Data.TensorDataset(test_x, test_y), batch_size=1000, shuffle=False)

epochs = 10
train_size = len(train_y)
test_size = len(test_y)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss().cuda()

tic = time.time()
for epoch in range(1, epochs + 1):
    total_train_loss = 0.
    train_nacc = 0.
    ts = time.time()
    for xi, yi in train_iter:
        xi, yi = xi.cuda(), yi.cuda()
        optimizer.zero_grad()
        logits = model(xi)
        loss = loss_fn(logits, yi)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_train_loss += loss.item() * len(xi)
            train_nacc += torch.sum(logits.argmax(axis=-1) == yi).item()
    train_cost, train_acc = total_train_loss / train_size, train_nacc / train_size

    total_test_loss = 0.
    test_nacc = 0.
    with torch.no_grad():
        for xi, yi in test_iter:
            xi, yi = xi.cuda(), yi.cuda()
            logits = model(xi)
            total_test_loss += loss_fn(logits, yi).item() * len(xi)
            test_nacc += torch.sum(logits.argmax(axis=-1) == yi).item()
    test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
    time_used = time.time() - ts
    print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# [1]TIME:10.604304552078247s COST:0.048627802077680825 ACC:0.94145 TEST_COST:0.048627802077680825 TEST_ACC:0.9833
# [2]TIME:10.243390798568726s COST:0.039858818612992765 ACC:0.9853166666666666 TEST_COST:0.039858818612992765 TEST_ACC:0.986
# [3]TIME:10.123138666152954s COST:0.02963086646050215 ACC:0.9900833333333333 TEST_COST:0.02963086646050215 TEST_ACC:0.9915
# [4]TIME:10.129415273666382s COST:0.028466253029182553 ACC:0.9931833333333333 TEST_COST:0.028466253029182553 TEST_ACC:0.9909
# [5]TIME:10.131487607955933s COST:0.02555006623733789 ACC:0.9945666666666667 TEST_COST:0.02555006623733789 TEST_ACC:0.9922
# [6]TIME:10.129435062408447s COST:0.03319597858935595 ACC:0.9952166666666666 TEST_COST:0.03319597858935595 TEST_ACC:0.9896
# [7]TIME:10.122684001922607s COST:0.02631072944495827 ACC:0.9958166666666667 TEST_COST:0.02631072944495827 TEST_ACC:0.9916
# [8]TIME:10.135545492172241s COST:0.027789497189223765 ACC:0.99605 TEST_COST:0.027789497189223765 TEST_ACC:0.9914
# [9]TIME:10.27768874168396s COST:0.027536340709775686 ACC:0.9978833333333333 TEST_COST:0.027536340709775686 TEST_ACC:0.9914
# [10]TIME:10.13521432876587s COST:0.030124087899457662 ACC:0.99685 TEST_COST:0.030124087899457662 TEST_ACC:0.9922
# 用时 102.0337393283844
