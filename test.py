import  paddle
import numpy as np

in1 = np.random.rand(256,256,128)
print(in1)
in2 = np.random.rand(256,128,128)
print(in1+in2)
x1 = paddle.to_tensor(in1)
x2 = paddle.to_tensor(in2)

out = paddle.concat([x1,x2], axis=1)
print(out.shape)