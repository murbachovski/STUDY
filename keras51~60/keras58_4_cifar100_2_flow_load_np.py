import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

save_path = 'd:/temp/'

x_train = np.load(save_path + 'keras58_cifar100_x_train.npy')
x_test = np.load(save_path + 'keras58_cifar100_x_test.npy')
y_train = np.load(save_path + 'keras58_cifar100_y_train.npy')
y_test = np.load(save_path + 'keras58_cifar100_y_test.npy')

#2. 모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D()) #(2,2)중 가장 큰 값 뽑아서 반의 크기(14x14)로 재구성함 / Maxpooling안에 디폴트가 (2,2)로 중첩되지 않도록 설정되어있음 
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')) 
model.add(Conv2D(12, 3))  #kernel_size=(2,2)/ (2,2)/ (2) 동일함 
model.add(MaxPooling2D())
model.add(Conv2D(filters=25, kernel_size=(3,3), padding='valid', activation='relu')) 
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(18, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='softmax')) 


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#3)fit
hist = model.fit(x_train,y_train, epochs=5,  # (fit_generator) x데이터,y데이터,batch_size까지 된 것
                    # steps_per_epoch=10,   # 훈련(train)데이터/batch = 160/5=32 (32가 한계사이즈임(max), 이만큼 잡아주는게 좋음/이상 쓰면 과적합, 더 적은 숫자일 경우 훈련 덜 돌게 됨)
                    # validation_data=[x_test, y_test],
                    batch_size = 300
                    # validation_steps=24,  # val(test)데이터/batch = 120/5=24
                    )  
#history=(metrics)loss, val_loss, acc
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print(acc) 
print("acc:",acc[-1])
print("val_acc:",val_acc[-1])
print("loss:",loss[-1])
print("val_loss:",val_loss[-1])

'''

'''