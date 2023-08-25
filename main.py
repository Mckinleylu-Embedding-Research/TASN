import random
import sys
sys.path.append("/root/Attibute_Social_Network_Embedding")
import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model import Graphormer
import wandb
from my_graphormer.data.my_dataset import CustomCora
import time

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置python环境种子
    np.random.seed(seed)
    torch.manual_seed(seed) # 设置CPU生成随机数的种子，方便下次复现实验结果，种子一样，每次运行的实验结果就会一样
    torch.cuda.manual_seed(seed) # 设置GPU生成随机数的种子，方便下次复现实验结果，种子一样，每次运行的实验结果就会一样
    torch.backends.cudnn.deterministic = True # 每次使用的卷积算法固定

def wandbSet():
    wandb.init(
    project= f'my_Graphormer',
    config={
    "lr": 1e-3,
     "epochs": 200,
     "device": device})
    
def train():
    model.train() # 训练模式
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    out = model(
                data.x.cuda(),
                data.edge_index.cuda(),
                data.edge_attr.cuda(),
                data.batch.cuda(),
            )
    # out = model(
    #             data.x,
    #             data.edge_index,
    #             data.edge_attr,
    #             data.batch,
    #         )

    loss = criterion(out, data.y.cuda())
    loss.backward()
    optimizer.step()
    return float(loss),out
  
@torch.no_grad()
def test():
    model.eval() # 模型开始测试
    pred = model(
                data.x.cuda(),
                data.edge_index.cuda(),
                data.edge_attr.cuda(),
                data.batch.cuda(),
            ).argmax(dim=-1)
    # pred = model(
    #             data.x,
    #             data.edge_index,
    #             data.edge_attr,
    #             data.batch,
    #         ).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs    
    

if __name__ == "__main__":
    seed_everything(42)
    root = "/root/Attibute_Social_Network_Embedding/my_graphormer/data/Cora/"
    data = CustomCora(root)
    data = DataLoader(data, batch_size=1, shuffle=False)
    for _ in data:
        data = _
        break
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    model = Graphormer(dim=64, head_num=4, layer_num=2).cuda()
    # model = Graphormer(dim=64, head_num=4, layer_num=2)
    data.to(device)
 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 采用交叉熵损失计算类别预测值和真实值之间的距离：


    
    #train_dataset = dataset[: int(len(dataset) * 0.7)]
    #test_dataset = dataset[int(len(dataset) * 0.7) :]
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    
    best_val_acc = final_test_acc = 0.0
    times = []
    for epoch in range(0, 30 + 1):
        start = time.time()
        loss,embedding = train() # 模型开始训练
        print('*'*30)
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        wandb.log({"Epoch":epoch, "Loss":loss, "Train":train_acc, "Val":val_acc, "Test":test_acc})
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    # for epoch in range(30):
    #     model.train()
    #     train_acc = 0.0
    #     train_loss = 0.0
    #     for data in dataset:
    #         optimizer.zero_grad()
    #         out = model(
    #             data.x.cuda(),
    #             data.edge_index.cuda(),
    #             data.edge_attr.cuda(),
    #             data.batch.cuda(),
    #         )
    #         loss = criterion(out, data.y.cuda())
    #         loss.backward()
    #         optimizer.step()
    #         pred = out.argmax(dim=1)
    #         train_acc += (pred == data.y.cuda()).sum() / len(data.y)
    #         train_loss += loss.item()

    #     model.eval()
    #     test_acc = 0.0
    #     test_loss = 0.0
    #     with torch.no_grad():
    #         for data in test_loader:
    #             out = model(
    #                 data.x.cuda(),
    #                 data.edge_index.cuda(),
    #                 data.edge_attr.cuda(),
    #                 data.batch.cuda(),
    #             )
    #             pred = out.argmax(dim=1)
    #             loss = criterion(out, data.y.cuda())
    #             test_acc += (pred == data.y.cuda()).sum() / len(data.y)
    #             test_loss += loss.item()
    #         test_acc = test_acc / len(test_loader)
    #         test_loss /= len(test_loader)
    #     print(
    #         f"Epoch: {epoch + 1:2d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    #     )
    #     wandb.log({"Epoch":epoch, "Train Loss":train_loss, "Train Acc":train_acc,"Test Loss":test_acc})        
    #     wandb.finish()