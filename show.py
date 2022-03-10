'''
绘制曲线、可视化
'''
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from CNN import CNN
import torch.nn as nn
import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets

def showPredict(model,data_loader):
    cnt = 0
    classes=['cardboard','glass','metal','paper','plastic','trash']
    for batch,(images,targets) in enumerate(data_loader):
        #forward
        outputs = model(images)
        maxnums,preds = torch.max(outputs,1)
        for idx,img in enumerate(images):
            ax = plt.subplot(3, 4, cnt+1)
            # ax.axis('off')
            img = img.detach().numpy()
            img = np.transpose(img,(1,2,0))
            ax.set_xlabel('label:%s\npredict:%s'%(classes[targets[idx]],classes[preds[idx]]))
            ax.imshow(img)
            
            if cnt==11:
                break
            cnt = cnt+1
        plt.tight_layout()
        plt.show()
        break

    

def showTrainLoss():
    file = open('train_losse-3','r')
    e_3 = file.readlines()
    file.close()
    file = open('train_losse-4','r')
    e_4 = file.readlines()
    file.close()
    file = open('train_losse-5','r')
    e_5 = file.readlines()
    file.close()
    file = open('train_losse-6','r')
    e_6 = file.readlines()
    file.close()
    
    e_3 = [float(i) for i in e_3]
    e_4 = [float(i) for i in e_4]
    e_5 = [float(i) for i in e_5]
    e_6 = [float(i) for i in e_6]
    epoch = [i for i in range(len(e_3))]
    
    plt.plot(epoch,e_3,label='1e-3')
    plt.legend()
    plt.show()
    plt.plot(epoch,e_4,label='1e-4')
    plt.plot(epoch,e_5,label='1e-5')
    plt.plot(epoch,e_6,label='1e-6')
    plt.legend()
    plt.show()

def showTrainAcc():
    file = open('train_accuracye-3','r')
    e_3 = file.readlines()
    file.close()
    file = open('train_accuracye-4','r')
    e_4 = file.readlines()
    file.close()
    file = open('train_accuracye-5','r')
    e_5 = file.readlines()
    file.close()
    file = open('train_accuracye-6','r')
    e_6 = file.readlines()
    file.close()
    
    e_3 = [float(i) for i in e_3]
    e_4 = [float(i) for i in e_4]
    e_5 = [float(i) for i in e_5]
    e_6 = [float(i) for i in e_6]
    epoch = [i for i in range(len(e_3))]
    
    plt.plot(epoch,e_3,label='1e-3')
    plt.plot(epoch,e_4,label='1e-4')
    plt.plot(epoch,e_5,label='1e-5')
    plt.plot(epoch,e_6,label='1e-6')
    plt.legend()
    plt.show()

def showTestAcc():
    file = open('test_accuracye-3','r')
    e_3 = file.readlines()
    file.close()
    file = open('test_accuracye-4','r')
    e_4 = file.readlines()
    file.close()
    file = open('test_accuracye-5','r')
    e_5 = file.readlines()
    file.close()
    file = open('test_accuracye-6','r')
    e_6 = file.readlines()
    file.close()
    
    e_3 = [float(i) for i in e_3]
    e_4 = [float(i) for i in e_4]
    e_5 = [float(i) for i in e_5]
    e_6 = [float(i) for i in e_6]
    epoch = [i for i in range(len(e_3))]
    
    plt.plot(epoch,e_3,label='1e-3')
    plt.plot(epoch,e_4,label='1e-4')
    plt.plot(epoch,e_5,label='1e-5')
    plt.plot(epoch,e_6,label='1e-6')
    plt.legend()
    plt.show()

def main():
    #数据集根地址
    path = "dataset/dataset-resized"
    #变换数据
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=200,
    shuffle=True, 
    drop_last=False)

    net = CNN(64,6)
    if os.path.exists('./modele-5.pth'):
        try:
            net.load_state_dict(torch.load('./modele-5.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    showPredict(net,dataloader)
    showTrainLoss()
    showTrainAcc()
    showTestAcc()

if __name__ == '__main__':
    main()