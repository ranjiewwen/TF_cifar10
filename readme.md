
## Cifar10 Tensorflow Project

### Get Started
- environment: `tensorflow-gpu1.8+cude9.0`
- datasets from kaggle : [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10/data), first you can download the train and test dataset.
- then use the `utils/get_data_list.py` and `utils/get_dataset_mean.py` scripts to generate `train.txt` and `val.txt`.

### How to Learn this project

- **one step:** you can modify trian parameters in `config/cifar10_config.json`.
- **two step:** you can learn how load datasets before training from `src/datasets/cifar10_dataloader.py`.
- **three step:** you can learn how to write network from `src/models/layers` and `src/models/simple_model.py`, you can easily create you own model.
- **four step:** you should finish trian scripts `tools/train_cifar10.py`, in this process you will finish loss function  and metric funtion：`src/loss/cross_entropy.py` and `src/metrics/acc_metric.py`; in this scripts `tools/train_cifar10.py`, we will first create graph and then run session. at the same time, we will record train models and use tensorboard to visual loss and accuracy in `experiments/chekpoint` and `experiment/summary` folder.
- **five step:** you can run train scripts:`tools/train_cifar10.py`.
- **six step:** when you get train model, you can predict image and get class name in `demo/prdict.py`.
![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/2808.png)
- **seven step:** you can also get some extra information from `demo/visual.py`, such as weights or visual feature map.
![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/featuremap.png)
- **other:** you can fimilar how to use some tool function in `tools/utils.py`.

## The optimization process

- The detailed information you can get from [there](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/scratch.md).
- run scripts `tools/trian_cifar10.py` **include adjust lr , add data augmentation ,add dropout ,weight decay,stack 3*3 conv training tricks. you can learn how train model acc from 70%+ to 91+%**, while add model depth through `conv4_1 and conv4_2` it can not imporve val acc.
![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/base_val_acc.png)
- run scripts `tools/trian_cifar10_v2.py` include add batch_norm, we can see it make the training more unstable, maybe it not imporve val acc, while stack 3*3 conv it can improve val acc remarkable.
![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/v2_val_acc.png)
- run scripts `tools/fintune_cifar10.py`. it frist load imagenet pretrain weights and then finetune resnet50.


## Reference

### finetune

- [tensorflow-cnn-finetune](https://github.com/dgurkaynak/tensorflow-cnn-finetune)
- [tensorflow-finetune-flickr-style](https://github.com/joelthchao/tensorflow-finetune-flickr-style)

### tiny-imagenet  

- [tiny_imagenet](https://github.com/search?q=tiny-imagenet&type=Repositories)

### mnist

- [mnist_with_summaries.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)
- [convolutional_network_raw.py](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py)

### other

- [cifar10-tensorflow](https://github.com/persistforever/cifar10-tensorflow)
- [CIFAR10_mxnet](https://github.com/yinglang/CIFAR10_mxnet)
- [models/tutorials/image/cifar10/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- [AutoAugment](https://github.com/DeepVoltaire/AutoAugment)
- [pytorch-auto-augment](https://github.com/4uiiurz1/pytorch-auto-augment)



