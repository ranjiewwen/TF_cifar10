
## Cifar10 Tensorflow Project

### Get Started
- environment: `tensorflow-gpu1.8+cude9.0`
- datasets from kaggle : [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10/data),first you can download the train dataset.
- then use the `utils/get_data_list.py` and `utils/get_dataset_mean.py` scripts, prepare for datsets `train.txt` and `val.txt`.

### How to Learn this project

- one step: you can modify trian parameters in `config/cifar10_config.json`.
- two step: you can learn how load data for before train from `src/datasets/cifar10_dataloader.py`.
- three step: you can learn how to write network from `src/models/layers` and `src/models/simple_model.py`, you can easy crate you own model.
- four step: you shoud finish trian scripts `tools/train_cifar10.py`, in this process you will finish `src/loss/cross_entropy.py` and `src/metrics/acc_metric.py`; in this scripts `tools/train_cifar10.py`, we will first create graph and then run session. at the some time, we will record train models and use tensorboard to visual loss and accuracy in `experiments/chekpoint` and `experiment/summary` folder.
- five step: you can run `tools/train_cifar10.py`.
- six step: when you get train model, you can predict image `demo/prdict.py`.
- seven step: you can also get some extra information from `demo/visual.py` such as weights or visual feature map.
- other: you can fimilar how to use some tool function in `tools/utils.py`.


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



