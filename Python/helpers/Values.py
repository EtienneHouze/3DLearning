#Contains constant values

import numpy as np

# Frequency for each label
freqs18 = np.array([0.32221022, 0.05355305, 0.19424014, 0.00857336, 0.0094591,  0.00856645, 0.00348288, 0.00707408, 0.13269694, 0.01248927, 0.0342964,  0.01020374, 0.00244545, 0.05837977, 0.0037118,  0.00186222, 0.00199306, 0.00090319, 0.13385889])

# Corresponidg weights
weights18 = 1/np.log(1.02+freqs18)

weights_dict = {i:weights18[i] for i in range(19)}