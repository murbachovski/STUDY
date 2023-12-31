import numpy as np
aaa = np.array(
    [2,3,4,5,6,7,8,-10,9,10,11,12,50]
) 

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위 : ', quartile_1) # 25% 4
    print('q2 : ', q2) # 50% 7
    print('3사분위 : ', quartile_3) # 75% 10
    iqr = quartile_3 - quartile_1 # 4분위수 6
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # -5
    upper_bound = quartile_3 + (iqr * 1.5) # 19
    return np.where((data_out>upper_bound) | (data_out<lower_bound)) # 19보다 크거나 -5보다 작은 넘
outliers_loc = outliers(aaa)
print('이상치의 위치 : ', outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()