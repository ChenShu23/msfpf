import argparse
import numpy as np
import pandas as pd
import torch, os, time
from DataLoader import Get_DataLoader
from model.network import Definition_Network
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



def Model_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Hyper_Parameter = train_parser()
    # Hyper Parameter 
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


    # test
    result = []
    result_original = []
    with torch.no_grad():
        Definition_model.eval()
        test_loss = 0
        for inflow_te in enumerate(data_test_loader):
            i_batch, (test_inflow_X, test_inflow_Y, test_inflow_Y_original, test_OD_time) = inflow_te
            # i_batch, (test_inflow_X, test_inflow_Y, test_inflow_Y_original) = inflow_te
            test_inflow_X, test_inflow_Y = test_inflow_X.type(torch.float32).to(device), test_inflow_Y.type(torch.float32).to(device)
            # test_OD_time = test_OD_time.type(torch.float32).to(device)
            target = Definition_model(test_inflow_X, test_OD_time, POI)
            loss = mse(input=test_inflow_Y, target=target)
            test_loss += loss.item()


            # evaluate on original scale
            # clone_prediction = target.cpu().detach().numpy().copy() * 1816  # clone(): Copy the tensor and allocate the new memory, * max_inflow(1816)  Dataset 1 = 1816  Dataset 2 = 1492
            clone_prediction = target.cpu().detach().numpy().copy() * 1492
            for i in range(clone_prediction.shape[0]):
                result.append(clone_prediction[i])

            # 获取result_original
            test_inflow_Y_original = test_inflow_Y_original.cpu().detach().numpy()
            # test_inflow_Y_original = test_inflow_Y_original.detach().numpy()
            for i in range(test_inflow_Y_original.shape[0]):
                result_original.append(test_inflow_Y_original[i])


        # print(np.array(result).shape, np.array(result_original).shape)  # (360, 64, pre_len)    (360, 64, pre_len)
        # 取整&非负取0
        result = np.array(result).astype(np.int64)
        result[result < 0] = 0
        result_original = np.array(result_original).astype(np.int64)
        result_original[result_original < 0] = 0

        # # Single station data
        x = [[] for index1 in range(result.shape[1])]
        y = [[] for index2 in range(result.shape[1])]
        for station in range(result.shape[1]):
            for i in range(result.shape[0]):
                x[station].append(result[i][station][0])
                y[station].append(result_original[i][station][0])
        # print(np.array(x).shape, np.array(y).shape)
        result = np.array(result).reshape(station_num, -1)
        result_original = result_original.reshape(station_num, -1)

        RMSE, R2, MAE, WMAPE = Metrics(result_original, result).evaluate_performance()

        avg_test_loss = test_loss / len(data_test_loader)
        print('test Loss:', avg_test_loss)



if __name__ == '__main__':
    Model_test()
