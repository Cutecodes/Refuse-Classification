from Dataset import getDataset
from CNN import CNN
import torch.nn as nn
import torch
import os

def train(model,optimizer,lossfunc,data_loader):
    total_loss = 0
    for batch,(images,targets) in enumerate(data_loader):
        #forward
        outputs = model(images)
        maxnums,preds = torch.max(outputs,1)
        loss = lossfunc(outputs,targets)
        #backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 batch * len(images), len(data_loader.dataset),
                100. * batch / len(data_loader), loss.item()))
    return total_loss/len(data_loader)

def test(model,data_loader):
    correct = 0
    total = 0
    for batch,(images,targets) in enumerate(data_loader):

        outputs = model(images)
        maxnums,preds = torch.max(outputs,1)
        correct += (preds == targets).sum().item()
        total+=targets.size(0)
    print("Accuracy:%s%%"%(100.*correct/total))
    return correct/total

def main():
    lossfunc = nn.CrossEntropyLoss()
    lr =0.0001
    batch_size = 200
    num_classes = 6
    image_size = 64
    epoch = 100 

    net = CNN(image_size,num_classes)
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainDataset,testDataset = getDataset()

    train_loader = torch.utils.data.DataLoader(
    trainDataset,
    batch_size=batch_size,
    shuffle=True, 
    drop_last=False)
    test_loader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=batch_size,
    shuffle=True)

    train_loss = open("train_loss","a")
    train_accuracy = open("train_accuracy","a")
    test_accuracy = open("test_accuracy","a")

    for i in range(epoch):
        print("train epoch:%s"%(i))
        tra_loss = train(net,optimizer,lossfunc,train_loader)
        train_loss.write("%s\n"%(tra_loss))
        train_acc = test(net,train_loader)
        train_accuracy.write("%s\n"%(train_acc))
        test_acc = test(net,test_loader)
        test_accuracy.write("%s\n"%(test_acc))
        torch.save(net.state_dict(),'./model.pth')
    train_loss.close()
    train_accuracy.close()
    test_accuracy.close()



if __name__ == '__main__':
    main()