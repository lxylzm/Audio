from Classifer.CLF import *
from time import time


feature = './Features/ship.csv'
metadata = './Features/meta.csv'

#Classifer(feature,metadata,model_name='NN',epoch=30,batch_size=512,leave_one_out=True,type=12)

Classifer(feature,metadata,model_name='SVM',leave_one_out=False,ten_k=True,type=2)
