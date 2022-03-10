# 任务分析
# 1、如果出现的序列是0000，那么下一位是0还是1显然不能确定
# 2、如果出现的序列是00001，那么下一位是1
# 3、如果序列是00001111，此时0和1的数量相同，显然这个序列下一步应该结束
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import numpy as np


# 首先构建RNN网络
class SimpleRnn(nn.Module):
    def __init__(self, input_size=4, hidden_size=2, output_size=3, num_layers=1):
        super(SimpleRnn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=False)  # batch_first=False:(seq(num_step), batch, input_dim)将序列长度放在第一位，batch放在第二位
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        '''
         基本运算过程：先进行embedding层的计算
         在将这个向量转换为一个hidden_size维度的向量。
         input的尺寸:(num_step,data_dim)
        '''
        x = self.embedding(inputs)  # x:(num_layers, batch_size, hidden_size)
        output, hidden = self.rnn(x, hidden)  # output:(num_step,batch_size,hidden_size)
        output = output[-1,:,:]  # output中包含了所有时间步的结果，从输出中选择最后一个时间步的数值,[batch_size,hidden_size]
        output = self.fc(output)  # [batch_size,output_size]
        return output, hidden

    def initHidden(self):  # 初始化隐藏层:(layer_size,batch_size,hidden_size)
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
# 生成数据
class Data():
    def __init__(self,sample=2000,max_n=10):
        self.sample = sample
        self.n = max_n
        # 定义n的不同权重，我们按照10:6:4:3:1:1....来配置n=1，2，3，4，5
        probablity = 1.0 * np.array([10, 6, 4, 3, 1, 1, 1, 1, 1, 1])
        # 保证n的最大值是sz，归一化成概率
        self.probablity = probablity[:self.n] / sum(probablity)
    def generateTrainData(self):
        # 1.生成sample个样本
        train_set = []
        for m in range(self.sample):
            # range生成序列，按probablity的概率分布抽样得输入序列长度n
            n = np.random.choice(range(1, self.n + 1), p=self.probablity)
            # 生成程度为2n这个字符串，用list的形式完成记录
            inputs = [0] * n + [1] * n
            # 在最前面插入3表示开始字符，在结尾插入2表示结束符
            train_set.append([3] + inputs + [2])
        return train_set
    def generateValidationData(self):
        # 2.生成sample/10的校验样本和少量长序列用于检验
        validion_set = []
        for m in range(self.sample // 10):
            n = np.random.choice(range(1, self.n + 1), p=self.probablity)
            inputs = [0] * n + [1] * n
            validion_set.append([3] + inputs + [2])
        for m in range(2):
            n = self.n + m
            inputs = [0] * n + [1] * n
            validion_set.append([3] + inputs + [2])
        return validion_set
# 开始训练
# 输入的size是4，可能的值为0,1,2(结束),3(结束)
# 输出size为3 可能为 0,1 2(结束)
class Trainer():
    def __init__(self, input_size=4, hidden_size=2, output_size=3,shuffletrain=True):
        self.Data = Data()
        self.traindata = self.Data.generateTrainData()
        self.validationdata = self.Data.generateValidationData()
        if shuffletrain:
            np.random.shuffle(self.traindata)
        self.rnn = SimpleRnn(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
        self.criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵函数,softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.001)

    def train(self,num_epoch):
        results = []
        for epoch in range(num_epoch):
            train_loss = 0
            # 对每一个序列进行训练
            for i, seq in enumerate(self.traindata,start=1):  # batch_size=1
                loss = 0
                hidden = self.rnn.initHidden()  # RNN特点！训练开始先初始化隐藏层
                # 对于每一个序列的所有字符进行循环
                for t in range(len(seq) - 1):
                    # 当前字符作为输入，下一个字符作为标签
                    x = Variable(torch.LongTensor([seq[t]]).unsqueeze(0))  # x的size为 time_steps=1,data_dimension = 1
                    y = Variable(torch.LongTensor([seq[t + 1]]))  # y的size data_dimension =1
                    output, hidden = self.rnn(x, hidden)  # output 的size：batch_size,output_size=3;  # hidden尺寸 layer_size = 1,batch_size = 1,hidden_size
                    loss += self.criterion(output, y)
                # 计算每一个字符的平均损失
                loss /= len(seq)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss

                # 打印结果
                if i % 500 == 0:
                    print('第{}轮，第{}个，训练平均loss：{:.2f}'.format(epoch, i, train_loss.data.numpy() / i))

            # 下面在校验集上进行测试
            validation_loss = 0
            errors = 0
            for i, seq in enumerate(self.validationdata,start=1):
                loss = 0
                outstring = ''
                label = ''
                diff = 0
                hidden = self.rnn.initHidden()  # 测试开始先初始化隐藏层！
                for t in range(len(seq) - 1):
                    x = Variable(torch.LongTensor([seq[t]]).unsqueeze(0))
                    y = Variable(torch.LongTensor([seq[t + 1]]))
                    output, hidden = self.rnn(x, hidden)
                    out = output.data.softmax(1).argmax()
                    # 以字符的形式添加到outputstring中
                    outstring += str(out.numpy())
                    label += str(y[0].data.numpy())
                    loss += self.criterion(output, y)  # 计算损失函数
                    # 输出模型预测字符串和目标字符串之间差异的字符数量
                    diff += (out!=y[0])
                loss /= len(seq)  # 平均误差
                validation_loss += loss
                errors += diff  # 计算累计的错误数
                if i % 40 ==0:  # 展示预测结果
                    print(f'sequence_{i}:{outstring}(predict) | {label}(label)')
            # 打印结果
            epoch_train_loss = train_loss.data.numpy() / len(self.traindata)
            epoch_validation_loss = validation_loss.data.numpy() / len(self.validationdata)
            epoch_uncorrectness = errors / len(self.validationdata)
            print('第{}轮，训练loss: {:.2f},校验loss：{:.2f},错误率：{:.2f}'.format(epoch,epoch_train_loss,epoch_validation_loss,epoch_uncorrectness))
            print('------------------------------------------------------------------------------------------------------')
            results.append([epoch_train_loss,epoch_validation_loss,epoch_uncorrectness])

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(num_epoch=30)