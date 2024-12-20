import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, auc
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


def plot_confusion_matrix(matrix, classes, savename):
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
           ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    #ax.set_xlabel('Predicted label')
    #ax.set_ylabel('True label')
    ax.set_xlabel('Diagnostic fault label',color='w')
    ax.set_ylabel('Real fault labels',color='w')
    plt.tick_params(axis='x',colors='w')
    plt.tick_params(axis='y',colors='w')
    #ax.patch.set_facecolor('greenyellow')
    fig.patch.set_facecolor('#004775')
    #plt.tick_params(axis='x',colors='w')
    #plt.tick_params(axis='y',colors='w')
    #plt.rcParams['figure.facecolor'] = 'b'
    plt.rcParams['savefig.dpi'] = 700 # 图片像素
    plt.rcParams['figure.dpi'] = 700 # 分辨率
    plt.savefig(savename, bbox_inches='tight' )

def huatu(Y_test1, result,classes):
    m=int(max(set(classes)))
    label_name=[]
    for i in range(1,m+1):
        num= str(i)
        str1='故障'
        name=str1+num
        label_name.append(name)
    plt.clf()
            #binary_sdae_result为模型测试输出(一列，类似于[0,1,3,2,1])，y_train为理想输出(一列)
    cm_CNNLSTM = confusion_matrix(result, Y_test1)
    plot_confusion_matrix(cm_CNNLSTM, label_name, '1.png')
    #plt.rcParams['figure.facecolor'] = 'b'
    plt.show()

def Plt_ROC(FPR, recall):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.plot(FPR, recall, c='w', label='ROC curve')  # ROC 曲线
    plt.plot(FPR, recall, c='greenyellow') 
    plt.title('ROC',color='w')  # 设置标题
    plt.plot([0,1],[0,1],'r--')
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR',color='w')
    plt.ylabel('Recall',color='w')
    #plt.legend(loc='lower right')
    #A=color(0,71,117)
    ax.patch.set_facecolor('#004775')
    fig.patch.set_facecolor('#004775')
    axe = plt.gca() 
    axe.spines['bottom'].set_color('w')
    axe.spines['left'].set_color('w')
    axe.spines['top'].set_color('w')
    axe.spines['right'].set_color('w')
    #axe.spines['left'].set_color('w')
    plt.tick_params(axis='x',colors='w')
    plt.tick_params(axis='y',colors='w')
    #plt.rcParams['figure.facecolor'] = 'b'
    plt.show()
    plt.rcParams['savefig.dpi'] = 700 # 图片像素
    plt.rcParams['figure.dpi'] = 700 # 分辨率
    plt.savefig('ROC.png' ,bbox_inches='tight' )
    
# TODO 绘制tsne图
def plot_tsne(X: np.ndarray, savename: str, Y=np.array([])):
    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    fig = plt.figure(figsize=(10, 10))  # 设置图像大小
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(4, -72)
    # 创建并运行t-SNE实例
    tsne = TSNE(n_components=3, perplexity=50, learning_rate=100)
    X_tsne = tsne.fit_transform(X)

    # 绘制t-SNE图

    # colors = ['red' if label == 1 else 'yellow' if label == 2 else 'blue' for label in Y]
    # for i, color in enumerate(colors):
    if len(Y) == len(X):
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=Y)
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])
    # plt.colorbar()
    plt.title('t-SNE visualization')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    
# classes
class M_CSA(nn.Module):
    def __init__(self, num_classes):
        super(M_CSA, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, padding=1) 
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1) 
        self.norm1 = nn.LayerNorm(14)  
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)  
        self.norm2 = nn.LayerNorm(64)  
        self.fc = nn.Linear(64, num_classes)  # 直接从64维特征映射到num_classes  


    def forward(self, x):
        # 假设x的维度是 [batch_size, in_channels, seq_length]
        # 通过第一个卷积层
        x = x.unsqueeze(1)  # 添加通道维，变为[batch_size, 1, sequence_length]  
        x = self.conv1(x) 
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm1(x)  
        x = F.relu(x)  
        
        N, C, L = x.size()  
        x = x.permute(2, 0, 1)  # (L, N, C) -> (L, N, C) for attention  
  
        attention_output, _ = self.attention(x, x, x)  
        attention_output = attention_output.permute(1, 2, 0)  # (L, N, C) -> (N, C, L)  
        attention_output = attention_output.mean(dim=2) 
        attention_output = self.norm2(attention_output)  
#         attention_output = F.relu(attention_output)  
        output = self.fc(attention_output)  # 使用正确的线性层名称  

        return output,attention_output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=10, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)