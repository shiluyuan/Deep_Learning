# Deep_Learning

## keras cnn
A simple example on MNIST data.

+ input layer : 784 nodes (MNIST images size)
+ first convolution layer : 5x5x32
+ first max-pooling layer
+ second convolution layer : 5x5x64
+ second max-pooling layer
+ third fully-connected layer : 1024 nodes
+ output layer : 10 nodes (number of class for MNIST)
+ for learning rate, I use 0.001

I run on tensorflow backend with GPU, the speed is quite fast!
My system: i7-7700k + 32GB 3000Mhz memory + GTX 1080 + Samsung 960 ssd
