import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

from sklearn.manifold import TSNE
from encode_model.m_csa_Model import *
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def train_m_csa(seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout, epochs, lr, gamma, train_DataFrame):
    martrix = train_DataFrame.values
    data = martrix[:, :-1]
    labels = martrix[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
    # 转换为 PyTorch 张量
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # train
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m_csa_model = M_CSA(num_classes=num_classes)
    vit_model = ViT(seq_len=seq_len,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout).to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
#     optimizer = optim.Adam(list(csa_model.parameters(), lr=lr)
    optimizer = optim.Adam(m_csa_model.parameters(), lr=lr)  
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    # scheduler
#     scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    all_epoch_losses = []  
    all_epoch_accuracies = []  
    all_epoch_val_losses = []  
    all_epoch_val_accuracies = [] 

    for epoch in range(epochs):
        true_labels = []
        pred_labels = []
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):

            data = data.to(device)
            label = label.to(device)

            output = m_csa_model(data)  # [16, 10, 10]
#             output = vit_model(output)  # [16, 3]

            label = label - 1
#             print(label.shape)
#             print(output.shape)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            
#             true_labels.append(label)
#             pred_labels.append(output.argmax(dim=1))

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = m_csa_model(data)
#                 val_output = vit_model(output)
                label = label - 1
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(test_loader)
                epoch_val_loss += val_loss / len(test_loader)

                true_labels.append(label)
                pred_labels.append(val_output.argmax(dim=1))
        
        all_epoch_losses.append(epoch_loss.item())  
        all_epoch_accuracies.append(epoch_accuracy.item())  
        all_epoch_val_losses.append(epoch_val_loss.item())  
        all_epoch_val_accuracies.append(epoch_val_accuracy.item())   
        
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

    # 保存模型
    torch.save(m_csa_model.state_dict(), './saved_model/ViT/m_csa_model_parameters.pth')
#     torch.save(vit_model.state_dict(), './saved_model/ViT/vit_model_parameters.pth')
    # TODO 从gpu取回数据
    if torch.cuda.is_available():
        true_labels = [true_labels[i].cpu().numpy() for i in range(0, len(true_labels))]
        pred_labels = [pred_labels[i].cpu().numpy() for i in range(0, len(pred_labels))]
    # TODO 计算最终的混淆矩阵
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    conf_mat = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(conf_mat, ['故障1', '故障2', '故障3'], './result/m_csa/confusion_matrix.png')
    # TODO 绘制训练数据tsne图
    tsne = TSNE(n_components=2, random_state=0)  
    X_tsne = tsne.fit_transform(X_test)  # 对测试数据进行t-SNE降维  
      
    plt.figure(figsize=(8, 6))  
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test)  # 使用真实标签进行着色  
    plt.colorbar(scatter)  
    plt.title('t-SNE visualization of test data')  
    plt.savefig('./result/m_csa/tsne_plot_test.png')  # 保存t-SNE图  
    plt.show() 
    
    return all_epoch_losses, all_epoch_val_losses, all_epoch_accuracies, all_epoch_val_accuracies


def test_m_csa(m_csa_model, epochs, criterion, val_loader, device):
    # val
    test_loader = val_loader
    m_csa_model.eval()
#     vit_model.eval()

    val_bar = tqdm(val_loader)

    epoch_val_accuracy = 0
    epoch_val_loss = 0
    with torch.no_grad():

        for data, label in val_bar:
            data = data.to(device)
            label = label.to(device)

            val_output = m_csa_model(data)
#             val_output = vit_model(output)
            label = label - 1
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)

    return epoch_val_loss, epoch_val_accuracy


def data_loader(data_path, label_true):
    test_DataFrame = pd.read_excel(data_path, header=None)
    martrix = test_DataFrame.values
    data = martrix[:, :]
    labels = np.array([label_true for _ in range(len(martrix[:, -1]))])
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
    # 转换为 PyTorch 张量
    X_test_tensor = torch.Tensor(data)
    y_test_tensor = torch.LongTensor(labels)
    # 创建数据集和数据加载器
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return test_loader


