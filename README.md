# Brevitas Fixed Point Training for Classification Models

This repo contains training scripts to train and evaluate popular deep learning classification models: Lenet, Alexnet and VGG using [Brevitas](https://github.com/Xilinx/brevitas).

## Requirements
- Pytorch >= 1.1.0
- Brevitas (https://github.com/Xilinx/brevitas)

## Install
After installing Pytorch, install Brevitas:
 
 ```bash
         git clone https://github.com/Xilinx/brevitas
         cd brevitas
         pip install .
 ```

Clone this repo

```bash
         git clone https://github.com/MinahilRaza/Brevitas_Fixed_Point.git
```

## Results

This repo includes pretrained models with the following accuracies

        | Name     | Dataset  | Weight quantization | Activation quantization | Brevitas Top1 |
        |----------|----------|---------------------|-------------------------|---------------|
        | Lenet    |  MNIST   |        8 bit        |         8 bit           |    98.99%     |
        | Lenet    |  MNIST   | 8 bit for conv only |   8 bit for conv only   |    99.08%     |
        | Alexnet  |  CIFAR10 |        8 bit        |         8 bit           |    85.42%     |
        | VGG      |  CIFAR10 |        8 bit        |         8 bit           |    86.22%     |

## Train

In order to launch training, run the following commands

### MNIST for Lenet

From within the *training_scripts* folder:
         ```bash
        python run.py --network Lenet --dataset MNIST
         ```

### AlexNet for CIFAR10

From within the *training_scripts* folder:
         ```bash
         python run.py --network AlexNet --dataset CIFAR10
         ```

### VGG for CIFAR10

From within the *training_scripts* folder:
         ```bash
         python run.py --network VGG --dataset CIFAR10
         ```

## Resuming Training and Evaluation

In order ro resume training from saved checkpoint use *--resume* flag. Set its value to true for resuming training.
For evaluation on validation set, set *--evaluate* flag to true.

## README format
This README format was inspired by brevitas_cnv_lfc (https://github.com/ussamazahid96/brevitas_cnv_lfc)
