from dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_graph(curve, name, save_path):
    plt.plot(curve, 'b')
    plt.xlabel('epochs')
    plt.ylabel(name)
    plt.savefig(save_path, bbox_inches='tight')
    return


def plot_confusion_matrix(matrix, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    fig.colorbar(cax)
    plt.savefig(save_path, bbox_inches='tight')
    return


def train_network(net, device, train_csv, val_csv, model_path):
    """
    @brief: Procedure to train and save neural network model.
    """
    #Reading the Kannada MNIST dataset
    train_images, train_labels, val_images, val_labels = read_data("train", train_csv, val_csv)
    train_set = KannadaDataset(train_images, train_labels, train_transforms())
    val_set = KannadaDataset(val_images, val_labels, test_transforms())
    print ("Data loaded.")

    #training hyperparameters
    lr = 0.001
    batch_size = 100
    max_epochs = 10

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    #train and save model with early stopping.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    train_accuracy_curve = list()
    loss_curve = list()
    val_accuracy_curve = list()
    best_val_accuracy = 0.

    for epoch in range(max_epochs):
        epoch_loss = 0.
        epoch_accuracy = 0.

        net.train()
        for i, (images, labels) in enumerate(train_data):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            epoch_accuracy += (predicted==labels).sum().item()
            epoch_loss += loss.item()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epochs, i+1, len(train_set)//batch_size, loss.item()))

        epoch_accuracy /= len(train_set)
        train_accuracy_curve.append(100*epoch_accuracy)
        loss_curve.append(epoch_loss)

        net.eval()
        with torch.no_grad():
            val_accuracy = 0.
            predictions = torch.LongTensor().to(device)

            for images, labels in val_data:
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                val_accuracy += (predicted==labels).sum().item()
                predictions = torch.cat((predictions, outputs.argmax(dim=1)), dim=0)

            val_accuracy /= len(val_set)
            val_accuracy_curve.append(100*val_accuracy)

            if (val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                best_predictions = predictions.cpu().numpy()
                torch.save(net.state_dict(), os.path.join(model_path, "model.ckpt"))

        print('Validation accuracy is: {} %'.format(100 * val_accuracy))

    #Plot training and validation curves
    plot_graph(train_accuracy_curve, 'training accuracy', os.path.join(model_path, "training_accuracy_graph.png"))
    plot_graph(loss_curve, 'training loss', os.path.join(model_path, "training_loss_graph.png"))
    plot_graph(val_accuracy_curve, 'validation accuracy', os.path.join(model_path, "validation_accuracy_graph.png"))

    #Plot the confusion matrix on validation set
    plot_confusion_matrix(confusion_matrix(np.array(val_labels), best_predictions), os.path.join(model_path, "confusion_matrix.png"))
    return
