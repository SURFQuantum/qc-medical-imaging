import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import h5py
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tqdm


class PatchCamelyon(data_utils.Dataset):

    def __init__(self, mode='train', batch_size=32, n_iters=None):
        super().__init__()

        self.n_iters = n_iters
        self.batch_size = batch_size

        assert mode in ['train', 'valid', 'test']
        base_name = "Camelyon_Patch/camelyonpatch_level_2_split_{}_{}.h5"

        print('\n')
        print("# " * 50)
        print('Loading {} dataset...'.format(mode))

        # Open the files
        self.h5X = h5py.File(os.path.join(base_name.format(mode, 'x')), 'r')
        self.h5y = h5py.File(os.path.join(base_name.format(mode, 'y')), 'r')

        print("# " * 50)
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        # size: 96x96x3
        image = np.array(self.h5X.get("x")[idx])
        # size: 1
        label = torch.tensor(self.h5y.get("y")[idx]).view(-1)

        image = self.transform(image)
        return image, label

    def __len__(self):
        # return len(self.X) // self.batch_size
        return len(self.h5X.get("x"))

classes = (0,1)

trainset = PatchCamelyon(mode='train', batch_size=256)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

testset = PatchCamelyon(mode='valid', batch_size=100)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)


# show images
#imshow(torchvision.utils.make_grid(images))


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in tqdm.tqdm(enumerate(trainloader, 0), total=1024):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels.squeeze(1))
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0
#
# print('Finished Training')
#
PATH = './camelyon.pth'
# torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
#imshow(torchvision.utils.make_grid(images))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in labels}
total_pred = {classname: 0 for classname in labels}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                #TODO:fix this shit with classes
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')