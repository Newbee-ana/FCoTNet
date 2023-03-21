# FCoTNet


# Required Environment
Ubuntu20.04  
GTX 2080Ti  
Python3.7  
PyTorch1.7.0  
CUDA10.2  
CuDNN7.0

# Usage Method(train with our dataset)
The model's backbone is ResNet. In our training, we use our 5-fold cross-validation dataset.  

```
# train and val
 python train.py --prefix FCoT --device 1 --epoch 100 
```
