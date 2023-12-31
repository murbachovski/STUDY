# DACNO DDARUNG
import numpy as np
from tensorflow.keras.models import Sequential     
from tensorflow.keras.layers import Dense               
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # = MSE
import pandas as pd # 전처리(CSV -> 데이터화)
from sklearn.preprocessing import LabelEncoder, RobustScaler

#1. DATA
path = './_data/wine/' # path ./은 현재 위치
# Column = Header

# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 
print(train_csv)
print(train_csv.shape) # (1459, 10)


# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 
print(test_csv)
print(test_csv.shape) # (715, 9)
print(train_csv.columns) #        'hour_bef_windspeed', 'hour_bef_humidity',    'hour_bef_visibility',
                         #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         #       dtype='object')
print(train_csv.info())
print(train_csv)        #[5497 rows x 13 columns]
print(test_csv.shape)   #(1000, 12)

le = LabelEncoder() # 정의 핏 트랜스폼
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa) #[1 0 1 ... 1 1 1]
print(type(aaa)) #<class 'numpy.ndarray'>
print(aaa.shape) #(5497,) 벡터형태
print(np.unique(aaa, return_counts=True)) #(array([0, 1]), array([1338, 4159], dtype=int64))

train_csv['type'] = aaa

test_csv['type'] = le.transform(test_csv['type'])
print(le.transform(['red', 'white'])) #[0 1]


# SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




# # ---  ------                  --------------  -----
# #  0   hour                    1459 non-null   int64
# #  1   hour_bef_temperature    1457 non-null   float64
# #  2   hour_bef_precipitation  1457 non-null   float64
# #  3   hour_bef_windspeed      1450 non-null   float64
# #  4   hour_bef_humidity       1457 non-null   float64
# #  5   hour_bef_visibility     1457 non-null   float64
# #  6   hour_bef_ozone          1383 non-null   float64
# #  7   hour_bef_pm10           1369 non-null   float64
# #  8   hour_bef_pm2.5          1342 non-null   float64
# #  9   count                   1459 non-null   float64
# # ---  ------                  --------------  -----

# print(train_csv.describe()) # [8 rows x 10 columns]

# print("TYPE",type[train_csv])

# ############################결측치 처리#############################
# #1. 결측치 처리 - 제거
# print(train_csv.isnull().sum())  #중요하답니다.
# train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv.shape) # (1328, 10)


# #########################################train_csv데이터에서 x와 y를 분리했다.
# #########이게 중요합니다#######
# x = train_csv.drop(['count'], axis = 1)
# print(x)
# y = train_csv['count']
# print(y)
# #########################################train_csv데이터에서 x와 y를 분리했다.

# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     shuffle=True,
#     train_size=0.7,
#     random_state=777
# )
# print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)   ---> (929, 9 ) (399, 9)
# print(y_train.shape, y_test.shape) # (1021, ) (438, )     ---> (929, ) (399, )
 
# #2. MODEL
# model = Sequential()
# model.add(Dense(1, input_dim = 9))

# #3. COMPILE
# model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train, y_train, epochs = 10, batch_size = 32, verbose=1)

# #4. EVALUATE, PREDICT
# loss = model.evaluate(x_test, y_test)
# print("loss:", loss)

