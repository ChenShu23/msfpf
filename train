import argparse
import numpy as np
import pandas as pd
import torch, os, time
from DataLoader import Get_DataLoader
from Network.Model import Definition_Network
from tensorboardX import SummaryWriter
from utils.earlystopping import EarlyStopping


def train_parser():
    parser = argparse.ArgumentParser(description="Parameter generation")
    parser.add_argument("--batch_size", type=int, required=True,
                        help='Training batch size')
    parser.add_argument("--lr", type=int, required=True,
                        help='Training learning rate')
    parser.add_argument("--hidden_layer", type=int, required=True,
                        help='Hidden layer in the model')
    parser.add_argument("--hidden_channel", type=int, required=True,
                        help='Hidden channel in the model')
    parser.add_argument("--pre_len", type=int, required=True,
                        help='Training prediction length')
    parser.add_argument("--epoch_num", type=int, required=True,
                        help='Training epoch number')
    parser.add_argument("--save_path", type=str, required=True,
                        help='Training epoch number')
    opt = parser.parse_args()
    return opt



def Model_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Hyper_Parameter = train_parser()
    # Hyper Parameter 超参数
    batch_size = Hyper_Parameter.batch_size
    lr = Hyper_Parameter.lr
    hidden_lay = Hyper_Parameter.hidden_layer
    hidden_cha = Hyper_Parameter.hidden_channel
    epoch_num = Hyper_Parameter.epoch_num
    save_path = Hyper_Parameter.save_path
    pre_len = Hyper_Parameter.pre_len

    # 数据读取
    Total_data = pd.read_csv('./Data/Inflow Data.csv', header=None, index_col=None, encoding="gbk")
    Total_data = Total_data.values
    POI = np.loadtxt('./Data/POI.csv', delimiter=",")
    POI = np.array([POI for i in range(batch_size)])
    POI = torch.tensor(POI)

    # Parameter
    time_lag = 10
    day_lag = 109
    station_num = Total_data.shape[0]
    N = 512

    # OD不固定
    OD_total = np.fromfile("./Data/OD_time.dat", dtype=np.int64, sep=",").reshape(day_lag*7*5, station_num, station_num)
    is_OD = True

    # dataloader
    data_train_loader, data_verify_loader, data_test_loader = \
        Get_DataLoader(Total_data, is_OD, OD_total, time_lag, pre_len, day_lag, batch_size)

    save_dir = './save_model/%s' % (save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Model
    global_start_time = time.time()
    writer = SummaryWriter()
    Definition_model = Definition_Network(time_lag, pre_len, station_num, device, N, hidden_cha, hidden_lay)
    Definition_model = Definition_model.to(device)
    optimizer = torch.optim.Adam(Definition_model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    temp_time = time.time()
    early_stopping = EarlyStopping(patience=150, verbose=False)
    print(Definition_model)


    # training
    for epoch in range(0, epoch_num):
        # model train
        train_loss = 0
        Definition_model.train()
        for inflow_tr in enumerate(data_train_loader):
            i_batch, (train_inflow_X, train_inflow_Y, train_OD_time) = inflow_tr
            train_inflow_X, train_inflow_Y = train_inflow_X.type(torch.float32).to(device), train_inflow_Y.type(torch.float32).to(device)
            train_OD_time = train_OD_time.type(torch.float32).to(device)
            predict = Definition_model(train_inflow_X, train_OD_time, POI)
            loss = mse(input=predict, target=train_inflow_Y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        with torch.no_grad():
            Definition_model.eval()
            val_loss = 0
            for inflow_val in enumerate(data_train_loader):
                i_batch, (val_inflow_X, val_inflow_Y, val_OD_time) = inflow_val
                val_inflow_X, val_inflow_Y = val_inflow_X.type(torch.float32).to(device), val_inflow_Y.type(torch.float32).to(device)
                target = Definition_model(val_inflow_X, val_OD_time, POI)
                loss = mse(input=val_inflow_Y, target=target)
                val_loss += loss.item()

        avg_train_loss = train_loss/len(data_train_loader)
        avg_val_loss = val_loss/len(data_train_loader)
        writer.add_scalar("loss_train", avg_train_loss, epoch)    # 保存训练时的erro值
        writer.add_scalar("loss_eval", avg_val_loss, epoch)       # 保存验证时的erro值
        writer.add_scalars("loss", {"avg_train_loss":avg_train_loss, "avg_val_loss":avg_val_loss}, epoch)
        print('epoch:', epoch, 'train Loss', avg_train_loss)

        # early stopping
        if epoch > 0:
            model_dict = Definition_model.state_dict()
            early_stopping(avg_val_loss, model_dict, Definition_model, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

        if epoch % 10 == 0:
            print("time for 10 epoches:", round(time.time() - temp_time, 2))
            temp_time = time.time()


    global_end_time = time.time() - global_start_time
    writer.close()
    print("global end time:", global_end_time)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    Model_train()
