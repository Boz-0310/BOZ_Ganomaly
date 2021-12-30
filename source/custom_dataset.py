# In[]
import os
# from keras.metrics import BCE
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

gray_flag= 0  #是否灰階
BCET_flag = 0  # 是否BCET
BGR_flag = 1 # 是否BGR
img_width =  128 #圖片統一size
img_height =  128 #圖片統一size
Train_num = 900  #正常照片訓練數
Test_num_pos = 100  # 正常照片測試數
Test_num_neg = 100  # 異常照片測試數

#資料維度 灰階[img_height, img_width] 或彩色[img_height, img_width, 3]
list_size = [img_height, img_width, (gray_flag-1)*-3]
if(0 in list_size):
    list_size.remove(0)


def BCET(img, E):
    l = np.min(img)  # 最小灰階值
    h = np.max(img)  # 最大灰階值
    e = np.mean(img)  # 平均灰階值
    L = 0  # 輸出最小灰階
    H = 255  # 輸出最大灰階
    E = E  # 輸出平均(自訂)
    N = np.shape(img)[0]*np.shape(img)[1]
    s = (np.sum(pow(img,2)))/N  # 圖片均方和

    B = (pow(h,2)*(E-L)-s*(H-L)+pow(l,2)*(H-E))/(2*(h*(E-L)-e*(H-L)+l*(H-E)))
    A = (H-L)/((h-l)*(h+l-2*B))
    C = L-A*(pow((l-B),2))

    Y = A*(pow((img-B),2))+C
    Y[Y > 255] = 255
    Y[Y < 0] = 0
    return Y


def get_data(train_path, gray_flag, BCET_flag, BGR_flag , img_width, img_height, label):
    img_path = glob(train_path + '/*.png') #抓取資料夾的圖片名稱
    nub_train = len(img_path)  # 抓取資料夾的圖片數量
    global list_size
    all_data = np.zeros([nub_train]+list_size, dtype=np.uint8) #建立DATA零矩陣
    all_y = np.zeros((nub_train,), dtype=np.uint8)  # 建立LABEL零矩陣
    for i in range(nub_train):
        #GRAY
        if gray_flag == 1:
            img = Image.open(img_path[i]).convert('L')  #讀取圖片且轉灰階
        else:
            img = Image.open(img_path[i])

        img = img.resize((img_width, img_height))  # 圖片resize
        arr_temp = np.asarray(img)  # 圖片轉array
        arr = np.zeros(np.shape(arr_temp))
        #BGR
        if BGR_flag == 1 and gray_flag == 0:
            arr[:,:,0] = arr_temp[:,:,2]
            arr[:,:,2] = arr_temp[:,:,0]
            arr[:,:,1] = arr_temp[:,:,1]
        else :
            arr = arr_temp 
        #BCET
        if BCET_flag == 1:
            if gray_flag == 1:
                arr = BCET(arr, 90)
            else:
                #三個顏色各自BCET再組合
                R = BCET(arr[:, :, 0], 90) 
                G = BCET(arr[:, :, 1], 90)
                B = BCET(arr[:, :, 2], 90)
                arr = np.dstack((R, G, B))
        all_data[i,...] = arr  # 賦值
        all_y[i] = label  
    return all_data, all_y


normal_x, normal_y = get_data(
   "./train/-Jpg", gray_flag, BCET_flag,BGR_flag, img_width, img_height, 1) #讀取全部正常照片
mura_x, mura_y = get_data(
   "./train/defect_png",  gray_flag, BCET_flag,BGR_flag, img_width, img_height, 0)  # 讀取全部異常照片
# normal_x, normal_y = get_data(
#     "./train_test/CC", gray_flag, BCET_flag, img_width, img_height, 0) #讀取全部正常照片
# mura_x, mura_y = get_data(
#     "./train_test/TT",  gray_flag, BCET_flag, img_width, img_height, 1)  # 讀取全部異常照片
np.random.shuffle(normal_x) #資料打亂
np.random.shuffle(mura_x) #資料打亂


x_Train = np.zeros([Train_num]+list_size, dtype=np.uint8)  #訓練資料零矩陣
x_Test = np.zeros([Test_num_pos+Test_num_neg]+list_size, dtype=np.uint8) #測試資料零矩陣

x_Test[0:Test_num_pos, ...] = normal_x[0:Test_num_pos, ...]
x_Test[Test_num_pos::, ...] = mura_x[0:Test_num_neg, ...]
y_Test = np.concatenate([normal_y[0:Test_num_pos],mura_y[0:Test_num_neg]])

#測試資料打亂
per = np.random.permutation(x_Test.shape[0])
new_x_Test = x_Test[per, ...]
new_y_Test = y_Test[per]


x_Train[0:Train_num, ...] = normal_x[Test_num_pos:Train_num+Test_num_pos, ...]
y_Train = normal_y[0:Train_num]


custom_train = (x_Train, y_Train)
custom_test=(new_x_Test,new_y_Test)
custom_data=(custom_train,custom_test)


# %%
