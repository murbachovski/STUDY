from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'

# 수치화
token = Tokenizer()
token.fit_on_texts([text1, text2]) # 두 개 이상은 리스트.
# print(token.word_index)    # 많은 놈이 앞의 인덱스를 가진다. 그 후 앞에서 순서대로...
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# print(token.word_counts)
# OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('지구용사', 1), ('배환희다', 1), ('멋있다', 1), ('또', 2), ('얘기해부아', 1)])

x = token.texts_to_sequences([text1, text2])
# print(x) # [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]] => (1, 11), (1, 7) => 1행 11열, 1행 7열, 길이가 다르다.
# print(type(x)) # <class 'list'>
# # 가치가 높다고 판단. => 원핫 해줘야합니다.

x = x[0] + x[1]
# print(x) # [2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13]

# ######  1. to_categorical  ######
from tensorflow.keras.utils import to_categorical

# x = to_categorical(x)
# print(x) # 불필요한 0이 생긴다.
# [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# print(x.shape)
# # (18, 14)


#######  2. get_dummies  ######
import pandas as pd
# x = pd.get_dummies(np.array(x).reshape(18, )) #넘파이로 바꿔준다 => # 1차원만 받습니다.
# # x = pd.get_dummies(np.array(x).ravel()) # => 쫙 펴서 1차원으로 만든다. ravel()
# # TypeError: unhashable type: 'list'
# # 1. 넘파이로 바꿔준다. 2. 왜 리스트를 받지 못할까?
# print(x)
# #     1   2   3   4   5   6   7   8   9   10  11  12  13
# # 0    0   1   0   0   0   0   0   0   0   0   0   0   0
# # 1    0   0   0   0   1   0   0   0   0   0   0   0   0
# # 2    0   0   1   0   0   0   0   0   0   0   0   0   0
# # 3    0   0   1   0   0   0   0   0   0   0   0   0   0
# # 4    0   0   0   0   0   1   0   0   0   0   0   0   0
# # 5    0   0   0   0   0   0   1   0   0   0   0   0   0
# # 6    0   0   0   0   0   0   0   1   0   0   0   0   0
# # 7    1   0   0   0   0   0   0   0   0   0   0   0   0
# # 8    1   0   0   0   0   0   0   0   0   0   0   0   0
# # 9    1   0   0   0   0   0   0   0   0   0   0   0   0
# # 10   0   0   0   0   0   0   0   0   1   0   0   0   0
# # 11   0   1   0   0   0   0   0   0   0   0   0   0   0
# # 12   0   0   0   0   0   0   0   0   0   1   0   0   0
# # 13   0   0   0   0   0   0   0   0   0   0   1   0   0
# # 14   0   0   0   0   0   0   0   0   0   0   0   1   0
# # 15   0   0   0   1   0   0   0   0   0   0   0   0   0
# # 16   0   0   0   1   0   0   0   0   0   0   0   0   0
# # 17   0   0   0   0   0   0   0   0   0   0   0   0   1
# print(x.shape) # (18, 13)

######  3. sklearn_onehot  ######
ohe = OneHotEncoder() # 2차원으로 받습니다.
x = ohe.fit_transform(np.array(x).reshape(-1, 1)).toarray()
print(x)
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
print(x.shape) # (18, 13)

# encoder = OneHotEncoder()
# x = np.array(x).reshape(-1, 1)
# x = encoder.fit_transform(x).toarray()
# print(x)
# # 결론은 내가 편한거 쓰면 됩니다.