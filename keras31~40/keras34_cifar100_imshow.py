from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)    
print(x_test.shape, y_test.shape)       
  
import matplotlib.pyplot as  plt
plt.imshow(x_train[49999], 'Blues')
plt.show()