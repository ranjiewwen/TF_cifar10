
## optimization process

- adjust learning rate

- exp1: learning rate exp_decay, note `decay_steps` sets.
- exp2: learning rate piecewise_constant sets.
- exp3: change piecewise_constant parameters.
- exp4: use sgd optimize.

```python
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

- data augmentation

```
def parse_aug_data(filename):

    I = np.asarray(cv2.imread(filename))
    I = I.astype(np.float32)
    mean = np.array([113.86538318359375,122.950394140625,125.306918046875])
    I -= mean

    if np.random.random() < 0.5 :
        I = cv2.flip(I,1)
    if np.random.random() < 0.5 :
        I = cv2.GaussianBlur(I, (3,3), 1.0)

    return I
```