# -*- coding: utf-8 -*-
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertForSequenceClassification
import os
import PySimpleGUI as sg
from torch.nn.functional import softmax
from collections import Counter


os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# 超参数设置
data_path = './data/train.json'  # 数据集
vocab_path = './data/vocab.pkl'  # 词表
# save_path = './saved_dict/lstmV2.ckpt'  # 模型训练结果
save_full_path = './saved_dict/lstmV2_full_checkpoint.ckpt'  # 完整状态保存路径
save_model_path = './saved_dict/lstmV2_model_weights.ckpt'   # 模型权重保存路径(推理用）
embedding_pretrained = \
    torch.tensor(
        np.load(
            './data/embedding_Tencent.npz')
        ["embeddings"].astype('float32'))
# 预训练词向量
embed = embedding_pretrained.size(1)  # 词向量维度
dropout = 0.3  # 随机丢弃
num_classes = 6  # 类别数
num_epochs = 15  # epoch数
batch_size = 128  # mini-batch大小
pad_size = 50  # 每句话处理成的长度(短填长切)
learning_rate = 1e-3  # 学习率
hidden_size = 128  # lstm隐藏层
num_layers = 2  # lstm层数
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def get_data():
    # tokenizer = lambda x: [y for y in x]  # 字级别
    # vocab = pkl.load(open(vocab_path, 'rb'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', timeout=60)  # 加载 BERT 分词器
    vocab = tokenizer.get_vocab()  # 使用 BERT 的词表
    # print('tokenizer',tokenizer)
    print('vocab', vocab)
    print(f"Vocab size: {len(vocab)}")

    # stop_words = load_stop_words('./data/stopWords.txt')

    def tokenize_and_convert_to_ids(text):
        # 先去除停用词
        # text = ''.join([word for word in text if word not in stop_words])
        # 使用 BERT 分词器分词并转换为 ID
        tokenized = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokenized)
        return token_ids


    train, dev, test = load_dataset(data_path, pad_size, tokenizer, vocab)
    # return vocab, train, dev, test

    # 提取所有标签用于计算权重
    all_labels = []
    for dataset_name, dataset in [("train", train), ("dev", dev), ("test", test)]:
        for _, label in dataset:
            if isinstance(label, (list, torch.Tensor)):  # 展平嵌套标签
                all_labels.extend(label if isinstance(label, list) else label.tolist())
            else:
                all_labels.append(label)
    print(f"get_data()--展平后的标签数量: {len(all_labels)}, 标签分布: {Counter(all_labels)}")

    # 提取所有标签用于计算权重
    # all_labels = [label for dataset in [train, dev, test] for _, label in dataset]
    return vocab, train, dev, test, all_labels


def load_stop_words(path):
    """
    加载停用词文件
    :param path: 停用词文件路径
    :return: 停用词列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f]
    return stop_words


def load_dataset(path, pad_size, tokenizer, vocab):
    """
    加载 JSON 格式数据集，处理停用词并转为 ID
    :param path: 数据集路径
    :param pad_size: 每个序列的最大长度
    :param tokenizer_function: 分词和转换 ID 的函数
    :param vocab: 词向量模型
    :param stop_words: 停用词列表
    :return: 训练集、验证集和测试集
    """
    contents = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            content, label = item
            tokens = tokenizer.tokenize(content)  # 分词
            # tokens = [token for token in tokens if not stop_words or token not in stop_words] # 去停用词
            tokens = ['[CLS]'] + tokens[:pad_size - 2] + ['[SEP]']  # 截断并添加特殊符号
            token_ids = tokenizer.convert_tokens_to_ids(tokens)  # 转为 ID
            # 处理 None 值，替换为 PAD（通常是0）
            token_ids = [id if id is not None else vocab.get(PAD, 0) for id in token_ids]
            seq_len = len(token_ids)
            if pad_size:
                token_ids.extend([vocab.get(PAD)] * (pad_size - seq_len))
                token_ids = token_ids[:pad_size]
            contents.append((token_ids, int(label)))
    # 数据集划分
    train, X_t = train_test_split(contents, test_size=0.4, random_state=42)
    dev, test = train_test_split(X_t, test_size=0.5, random_state=42)
    # 打印每个数据集的样本数和标签分布
    for split_name, dataset in [("train", train), ("dev", dev), ("test", test)]:
        labels = [label for _, label in dataset]
        print(f"{split_name} 数据集: 样本数={len(dataset)}, 标签分布={Counter(labels)}")

    return train, dev, test


# get_data()

class TextDataset(Dataset):
    def __init__(self, data, pad_token=0):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        # self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
        # 清理 None 值
        # 替换 None 为 pad_token（默认为 0）
        self.x = torch.LongTensor([
            [pad_token if id is None else id for id in x[0]]  # 替换 None 为 pad_token
            for x in data
        ]).to(self.device)

        self.y = torch.LongTensor([x[1] if x[1] is not None else 0 for x in data]).to(self.device)

    def __getitem__(self, index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label

    def __len__(self):
        return len(self.x)


# 以上是数据预处理的部分

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 定义LSTM模型
class LSTMWithBERTEmbedding(torch.nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers, dropout):
        super(LSTMWithBERTEmbedding, self).__init__()
        # 加载 BERT 模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 冻结/解冻 BERT 参数（False/True）
        for param in self.bert.parameters():
            param.requires_grad = True

        # 解冻 BERT 最后一层参数
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = 'layer.11' in name or 'pooler' in name or 'cls' in name

        # # 解冻部分 BERT 参数
        # for name, param in model.bert.named_parameters():
        #     if 'encoder.layer.8' in name or 'encoder.layer.9' in name:
        #         param.requires_grad = True  # 解冻第 8 层及以上层参数

        # 定义 LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.bert.config.hidden_size,  # 嵌入维度与 BERT 输出对齐
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # 分类器
        self.fc = torch.nn.Linear(hidden_size * 2 * 2, num_classes)  # 双向 LSTM 输出维度翻倍，且池化增加维度

    def forward(self, input_ids, attention_mask):
        # with torch.no_grad():  # 不使用 torch.no_grad，让 BERT 的参数在训练时更新
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask) # BERT 提取序列嵌入
        embeddings = bert_output.last_hidden_state  # 使用最后一层隐状态作为输入
        lstm_output, _ = self.lstm(embeddings)  # 经过 LSTM 层

        # 新的池化方法：结合全局最大池化和全局平均池化
        max_pooled = torch.max(lstm_output, dim=1).values  # 最大池化
        mean_pooled = torch.mean(lstm_output, dim=1)  # 平均池化

        # 拼接池化结果
        pooled_output = torch.cat([max_pooled, mean_pooled], dim=1)

        # 分类器
        logits = self.fc(pooled_output)
        # logits = self.fc(lstm_output[:, -1, :])  # 使用最后一个时间步的输出进行分类
        return logits


# 权重初始化，默认xavier
# xavier和kaiming是两种初始化参数的方法
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                # 只有在权重是二维以上时，才应用xavier初始化
                if w.dim() >= 2:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)


def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)

def calculate_class_weights(labels, num_classes):
    flattened_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            flattened_labels.extend(label.flatten().tolist())
        else:
            flattened_labels.append(label)

    labels = flattened_labels
    labels = [int(label) for label in labels]

    # 统计类别数量
    class_counts = [0] * num_classes
    total_samples = len(labels)

    print(f"calculate_class_weights()--展平后的标签数量: {len(labels)}, 标签分布: {Counter(labels)}")  # 调试打印

    # 统计每个类别的样本数量
    for label in labels:
        class_counts[label] += 1

    # 计算类别权重
    class_weights = []
    for c in range(num_classes):
        if class_counts[c] == 0:
            print(f"警告: 类别 {c} 没有样本！")
            class_weights.append(0)
        else:
            class_weights.append(total_samples / (num_classes * class_counts[c]))

    return torch.tensor(class_weights, dtype=torch.float)


def collate_fn(batch):
    # print("Start collate_fn")
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # print(f"Raw labels: {labels}")  # 检查标签的原始内容
    max_len = max(len(ids) for ids in input_ids)

    # Padding input_ids 和 attention_mask
    padded_input_ids = []
    attention_masks = []
    for ids in input_ids:
        # 替换 None 为 pad_token（通常是 0）
        padded_ids = [id if id is not None else 0 for id in ids] + [0] * (max_len - len(ids))
        attention_mask = [1 if id != 0 else 0 for id in padded_ids]  # 0作为填充token
        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)

    # print("End collate_fn")
    # print(f"Batch input shape: {len(padded_input_ids)}, Label shape: {len(labels)}")  # 调试打印
    return torch.tensor(padded_input_ids), torch.tensor(attention_masks), torch.tensor(labels)


# 定义训练的过程
def train(model, dataloaders, num_classes):
    '''
    训练模型
    :param model: 模型
    :param dataloaders: 处理后的数据，包含trian,dev,test
    '''
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # train_labels = []
    # for batch in dataloaders['train']:
    #     # 打印batch的结构，查看标签的位置
    #     print(batch)
    #
    #     # 如果是元组格式
    #     labels = batch[2]  # 或根据实际结构修改
    #
    #     # 如果是字典格式
    #     # labels = batch['labels']
    #
    #     train_labels.extend(labels.cpu().numpy())

    # 获取训练集的标签用于计算类别权重
    train_labels = []
    for batch in dataloaders['train']:
        labels = batch[2]  # 假设标签是batch的第三个元素
        train_labels.extend(labels.cpu().numpy())  # 将标签添加到train_labels中

    # 计算类别权重（仅基于训练集的标签）
    class_weights = calculate_class_weights(train_labels, num_classes)

    # 使用加权交叉熵损失函数
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 修改优化器与学习率
    # 原optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.lstm.parameters(), 'lr': 1e-3},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])

    # 增加学习率调度器：每 5 个 epoch 后学习率减小为原来的 0.1 倍
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf') # 初始化最佳验证损失
    early_stop_count = 0
    max_early_stop = 3  # 早停计数器


    print("Start Training...\n")
    plot_train_acc = []
    plot_train_loss = []

    for epoch in range(num_epochs):
        # 1，训练循环----------------------------------------------------------------
        # 将数据全部取完
        # 记录每一个batch

        # 训练模式，可以更新参数
        model.train()
        step = 0
        train_lossi = 0
        train_acci = 0
        # print("进入dataloaders...\n")
        print(f"Total batches in dataloaders['train']: {len(dataloaders['train'])}")
        print(f"Epoch {epoch + 1} Training...")

        for step, batch in enumerate(dataloaders['train']):
            # if step >= 10: # 只跑10个batch
            #     break

            # inputs, attention_masks, labels = batch
            # print(f"Step {step}: Inputs shape={inputs.shape}, Labels shape={labels.shape}")
            # print(f"Sample labels: {labels[:5]}")  # 查看前几个标签
            # break

            if step % 5 == 0:  # 每处理5个batch输出一次调试信息
                print(f"Processing batch {step + 1}/{len(dataloaders['train'])}...")

            # print("Batch loaded...")
            inputs, attention_masks, labels = batch  # 从数据加载器中解包
            # print(f"Batch {step + 1}: inputs.shape={inputs.shape}, labels.shape={labels.shape}")
            inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)
            # print("Moved to device...")

            # 梯度清零，防止累加
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_masks)
            # print("Model forward pass complete...")

            loss = loss_function(outputs, labels)
            # print(f"Loss computed: {loss.item()}")

            loss.backward()
            # print("Backward pass complete...")

            optimizer.step()
            # print("Optimizer step complete...")

            # step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)

        print("一轮训练的batch都处理完成，退出dataloaders...\n")
        scheduler.step()  # 学习率调度器更新
        print("Scheduler step complete...")
        # 2，验证集验证----------------------------------------------------------------
        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function, Result_test=False)
        train_acc = train_acci / (step + 1)
        train_loss = train_lossi / (step + 1)
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        print("epoch = {} :  train_loss = {:.3f}, train_acc = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}".
              format(epoch + 1, train_loss, train_acc, dev_loss, dev_acc))

        # 保存模型权重和完整状态（包括优化器状态）
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            early_stop_count = 0
            # 保存完整训练状态
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dev_best_loss': dev_best_loss,
            }, save_full_path)
            # 保存仅模型权重
            torch.save(model.state_dict(), save_model_path)
        else:
            early_stop_count += 1
            if early_stop_count >= max_early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break


    # 绘制训练曲线
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)
    # 3，验证循环----------------------------------------------------------------
    # 加载最佳模型进行测试
    checkpoint = torch.load(save_full_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function, Result_test=True)
    elapsed_time = time.time() - start_time

    print('================' * 8)
    print(f'Test: test_loss={test_loss:.3f}, test_acc={test_acc:.2%}')
    print(f'Test evaluation completed in {elapsed_time:.2f} seconds.')


def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='micro')
    recall = recall_score(real, pred, average='micro')
    f1 = f1_score(real, pred, average='micro')
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1,))

    # 设置中文字体 C:\Windows\Fonts\msyh.ttc
    font_path = "C:/Windows/Fonts/msyh.ttc"
    my_font = fm.FontProperties(fname=font_path)

    labels11 = ['其他', '喜好', '悲伤', '厌恶', '愤怒', '高兴']
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='', ax=ax)

    # 显式设置标签字体
    ax.set_xticklabels(labels11, fontproperties=my_font)
    ax.set_yticklabels(labels11, fontproperties=my_font)

    # 设置标签的中文字体
    plt.xlabel("预测值", fontproperties=my_font)
    plt.ylabel("真实值", fontproperties=my_font)
    plt.title("混淆矩阵", fontproperties=my_font)

    plt.savefig("results/reConfusionMatrixCN.tif", dpi=400)
    plt.show()

# 模型评估
def dev_eval(model, data, loss_function, Result_test=False):
    '''
    :param model: 模型
    :param data: 验证集集或者测试集的数据
    :param loss_function: 损失函数
    :return: 损失和准确率
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    batch_count = 0  # 计数器，用来限制最大 batch 数量

    with torch.no_grad():
        for inputs, attention_masks, labels in data:
            # 限制最大处理的 batch 数量
            # if batch_count >= 10:
            #     break

            # 调试输出只每处理几个batch一次 {len(dataloaders['train'])}
            if batch_count % 5 == 0:
                print(f"Evaluating batch {batch_count + 1}/{len(data)}...")

            inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)

            outputs = model(inputs, attention_mask=attention_masks)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            batch_count += 1  # 每处理一个 batch 计数器加 1

    acc = metrics.accuracy_score(labels_all, predict_all)
    if Result_test:
        result_test(labels_all, predict_all)
    return acc, loss_total / len(data)


# 训练模型
# if __name__ == '__main__':
#     # 设置随机数种子，保证每次运行结果一致，不至于不能复现模型
#     np.random.seed(1)
#     torch.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     torch.backends.cudnn.deterministic = True  # 保证每次结果一样
#
#     start_time = time.time()
#     print("Loading ...")
#
#     # 加载停用词
#     # stop_words = load_stop_words('./data/stopWords.txt')
#
#     # 加载数据集
#     print("Loading data with BERT vocab...")
#     vocab, train_data, dev_data, test_data, all_labels = get_data()
#
#     print(f"Sample vocab items: {list(vocab.keys())[:10]}")  # 打印词表的前10个词
#     print(f"Vocab size: {len(vocab)}")  # 打印词表大小
#
#     print(f"train_data sample: {train_data[:3]}")  # 查看训练数据的前几条
#     print(f"dev_data sample: {dev_data[:3]}")  # 查看验证数据的前几条
#     print(f"test_data sample: {test_data[:3]}")  # 查看测试数据的前几条
#
#     dataloaders = {
#         'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True, collate_fn=collate_fn),
#         'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=True, collate_fn=collate_fn),
#         'test': DataLoader(TextDataset(test_data), batch_size, shuffle=True, collate_fn=collate_fn)
#     }
#
#     # 检查一个 batch 的输出
#     print("Checking DataLoader...")
#     for batch in dataloaders['train']:
#         input_ids, attention_mask, labels = batch
#         print(f"Sample input_ids: {input_ids[0]}")
#         print(f"Sample attention_mask: {attention_mask[0]}")
#         print(f"Sample label: {labels[0]}")
#         print(
#             f"Input shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}, Labels shape: {labels.shape}")
#         break
#
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model = LSTMWithBERTEmbedding(hidden_size=hidden_size,
#                                   num_classes=num_classes,
#                                   num_layers=num_layers,
#                                   dropout=dropout).to(device)
#
#     print("Testing model forward pass...")
#     for batch in dataloaders['train']:
#         input_ids, attention_mask, labels = batch
#         input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
#
#         # 测试模型输出
#         outputs = model(input_ids, attention_mask)
#         print(f"Output logits shape: {outputs.shape}")  # 输出形状应该为 [batch_size, num_classes]
#         print(f"Sample logits: {outputs[0]}")
#         break
#
#     init_network(model)
#     train(model, dataloaders, num_classes)


# 直接加载模型并生成混淆矩阵
# if __name__ == '__main__':
#     # 设置随机数种子，保证每次运行结果一致
#     np.random.seed(1)
#     torch.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     torch.backends.cudnn.deterministic = True  # 保证每次结果一样
#
#     start_time = time.time()
#     print("Loading test data ...")
#
#     # 加载停用词
#     stop_words = load_stop_words('./data/stopWords.txt')
#     # 加载数据集
#     vocab, train_data, dev_data, test_data, all_labels = get_data()
#
#     # 准备测试集 DataLoader
#     test_dataloader = DataLoader(TextDataset(test_data), batch_size, shuffle=False, collate_fn=collate_fn)
#
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model = LSTMWithBERTEmbedding(hidden_size=hidden_size,
#                                   num_classes=num_classes,
#                                   num_layers=num_layers,
#                                   dropout=dropout).to(device)
#
#     # 加载模型权重
#     model.load_state_dict(torch.load(save_model_path))
#     model.eval()  # 设置为评估模式
#     print("Model loaded successfully.")
#
#     # 测试并生成混淆矩阵
#     loss_function = torch.nn.CrossEntropyLoss()
#     print("Generating confusion matrix ...")
#     dev_eval(model, test_dataloader, loss_function, Result_test=True)
#
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)


###################### 以下为带可视化界面部分的注释 ######################

# 模型预测函数
def predict_text(text, model, tokenizer, max_len, device):
    """
    使用训练好的模型对文本进行预测
    :param text: 输入的文本
    :param model: 加载的训练好的模型
    :param tokenizer: BERT 分词器
    :param max_len: 文本最大长度
    :param device: 设备 (CPU/GPU)
    :return: 分类标签
    """
    model.eval()
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = softmax(outputs, dim=1).cpu().numpy()
    pred = np.argmax(probs, axis=1)[0]
    labels = ['其他', '喜好', '悲伤', '厌恶', '愤怒', '高兴']  # 根据您的模型定义
    return labels[pred]

# 图形界面主函数
def main_gui(model, tokenizer, max_len, device):
    sg.theme("LightBlue")  # 设置主题

    # 定义按钮的颜色
    button_color = ('white', '#87CEEB')  # 字体颜色为白色，背景色为深蓝色
    button_font = ("Helvetica", 15, "bold")  # 粗体字体

    # 定义菜单栏和布局
    menu_def = [['Help', ['About...']]]
    layout = [
        [sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
        [sg.Text('请输入文本进行情感分析:', font=("Helvetica", 15))],
        [sg.Multiline(s=(60, 10), key='_INPUT_TEXT_', expand_x=True, background_color='#e9ecef', text_color='black', font=("Helvetica", 18))],
        [sg.Text('分析结果：', font=("Helvetica", 15)), sg.Text('     ', key='_OUTPUT_', font=("Helvetica", 15))],
        [
            sg.Button('开始', font=button_font, button_color=button_color),
            sg.Button('清空', font=button_font, button_color=button_color)
        ]
    ]

    # 创建窗口
    window = sg.Window(
        '情感分析系统',
        layout,
        resizable=True,
        finalize=True,
        keep_on_top=True
    )

    while True:
        event, values = window.read()

        if event in (None, 'Exit'):
            break
        elif event == '开始':
            input_text = values['_INPUT_TEXT_'].strip()
            if input_text:
                result = predict_text(input_text, model, tokenizer, max_len, device)
                window['_OUTPUT_'].update(result)
            else:
                window['_OUTPUT_'].update('请输入有效文本')
        elif event == '清空':
            window['_INPUT_TEXT_'].update('')
            window['_OUTPUT_'].update('')

    window.close()


# 主函数（带可视化界面）
if __name__ == "__main__":
    # 设备设置
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', timeout=60)
    model = LSTMWithBERTEmbedding(
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(torch.load(save_model_path))  # 加载模型权重
    model = model.to(device)  # 模型迁移到设备

    # 启动界面
    print("Launching GUI...")
    main_gui(model, tokenizer, max_len=pad_size, device=device)


