import model
import torch
from torch import nn
from torch import optim

class trainer:
    def __init__(self):
        device = torch.device('cpu')

        self.net = model.ConvNet()
        self.net.to(device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.n_epochs = 10

    def train(self,trainloader,testloader):
        self.net.train()
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            print_every = 200  # mini-batches
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # Transfer to GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i % print_every) == (print_every-1):
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
                    running_loss = 0.0


            # Print accuracy after every epoch
            accuracy = self.compute_accuracy(self.net, testloader)
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))

        print('Finished Training')

    def compute_accuracy(self,net, testloader):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total