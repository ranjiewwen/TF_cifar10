
## optimization process

### adjust learning rate

- exp1: learning rate exp_decay, note `decay_steps` sets.
- exp2: learning rate piecewise_constant sets.
- exp3: change piecewise_constant parameters.
- exp4: use sgd optimize.

```
"lr": 0.001,
def get_lr_strategy(config,global_step):

    if config.lr_plan == "exp_decay":
        learning_rate = tf.train.exponential_decay(learning_rate=config.lr,
                                           global_step=global_step,
                                           decay_rate=0.9,
                                           decay_steps=1800,
                                           staircase = True
                                           )
    elif config.lr_plan == "piecewise":
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=[int(config.max_iter * 0.6), int(config.max_iter * 0.8)],
                                                    values=[config.lr, config.lr * 0.2, config.lr * 0.02])
    return learning_rate
```

### experiments results:

- tensorboard visual learning rate.

![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/lr.png)

-  tensorboard visual val accuracy.

![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/lr_acc.png)
![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/val_acc_.png)

- we can see exp_decay and sgd optimize is not a good choice, while use piecewise_constant learning rate can Increase accuracy, **val_accuracy achieve 74+%**.

### data augmentation

- exp1: baseline use piecewise_constant learning rate.
- exp2: use flip and GaussianBlur data augmentation.

```
    mean = np.array([113.86538318359375,122.950394140625,125.306918046875])
    I -= mean

    if np.random.random() < 0.5 :
        I = cv2.flip(I,1)
    if np.random.random() < 0.5 :
        I = cv2.GaussianBlur(I, (3,3), 0.5)
```

- exp3: use add_gasuss_noise and image_whitening.

```
# https://feelncut.com/2018/09/11/182.html
def add_gasuss_noise(image, mean=0, var=0.001):
    '''
        mean : 均值
        var : 方差
    '''
    image = np.array(image, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)

    return out

def image_whitening(image):

    mean = np.mean(image)
    std = np.max([np.std(image),1.0/np.sqrt(image.shape[0]*image.shape[1]*image.shape[2])])
    white_image = (image-mean)/std

    return white_image

def parse_aug_data(filename):

    I = np.asarray(cv2.imread(filename))
    I = I.astype(np.float32)

    # mean = np.array([113.86538318359375,122.950394140625,125.306918046875])
    # I -= mean
    I = image_whitening(I)

    if np.random.random() < 0.5 :
        I = cv2.flip(I,1)
    if np.random.random() < 0.5 :
        I = cv2.GaussianBlur(I, (3,3), 0.5)
    if np.random.random() < 0.5:
        I = add_gasuss_noise(I)

    return I
```

### experiments results:

- tensorboard visual cross_entropy loss.

![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/aug_loss.png)

-  tensorboard visual val accuracy.

![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/aug_acc.png)

- there we can see add data augmentation can increase val accuracy, when use image_whitening can speed up convergence. at last **val_accuracy achieve 81+%**.

### model adjust

- exp1: add dropout, train and test keep_prob is not same.
```
sess.run(train_step, feed_dict = {x:train_batch,y:label_batch,dropout_keep_prob: config.keep_prob})  

val_acc = sess.run(accuracy,feed_dict={x:val_batch,y:val_label,dropout_keep_prob: 1.})

```

- exp2: add weight decay(conv and fc decay coef is same important)
```
def conv(input, out_channel,weight_decay = 0.0001, trainable=True):

def fc(input, out_channel,weight_decay = 0.01,trainable=True):

```

- exp3: add image crop augmentation
```
def image_crop(image):

    image = np.pad(image,[[4,4],[4,4],[0,0]],'constant')
    left = np.random.randint(image.shape[0]-32+1)
    top = np.random.randint(image.shape[1]-32+1)
    ret_img = image[left:left+32,top:top+32,:]

    return ret_img
```

- exp4: add batch normalization and drop dropout
```markdown
 bn1 = batch_norm_wrapper(relu1,self.is_training,self.config.moving_ave_decay,self.config.UPDATE_OPS_COLLECTION)
```

- exp5: add bn and increase network depth
``` 
        with tf.variable_scope("conv4"):
            conv4 = conv(pool3,256)
            relu4 = tf.nn.relu(conv4)
            bn4 = batch_norm_wrapper(relu4, self.is_training,self.config.moving_ave_decay,self.config.UPDATE_OPS_COLLECTION)
        pool4 = maxpool("pool3",bn4)
```


### experiments results:

- tensorboard visual cross_entropy+regularization_losses.

![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/mdoel_loss.png)

-  tensorboard visual val accuracy.

![](https://github.com/ranjiewwen/TF_cifar10/blob/master/doc/image/model_acc.png)

- we can see dropout get acc 83+%, weight decay get acc 84+%, image_crop augmentation get amazing acc 88+%, when add bn(exp4) while train not stable, maybe not good adjust parameter, acc 88+%, when add one more conv4(exp5), training is bad. next will continue optimize this model.
