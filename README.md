# machine_learning

### Required

1. python 3.6.8

2. CUDA 9.0.176

3. CuDNN 7.0.5

4. tensorflow-gpu==1.10.0

5. install this requirements:

matplotlib==3.0.2

Keras==2.2.4

pandas==0.24.1

numpy==1.16.1

scikit_learn==0.20.2

```
pip install -r requirements.txt
```

### record and update logs in "tensorboard"
```
python -m tensorboard.main --logdir=tenzor_logs/
```

### results:

**Model: logistics_regression_model, elapsed time: DAYS:0 HOURS:0 MIN:2 SEC:30, validation accuracy: 0.851**

**Model: 5_layers_adadelta_optim_model, elapsed time: DAYS:0 HOURS:0 MIN:11 SEC:58, validation accuracy: 0.895**

**Model: new_convolutional_model, elapsed time: DAYS:0 HOURS:0 MIN:29 SEC:56, validation accuracy: 0.930**

**Model: convolutional_model, elapsed time: DAYS:0 HOURS:0 MIN:16 SEC:49, validation accuracy: 0.908**

**Model: batch_normalization_convolutional_model, elapsed time: DAYS:0 HOURS:0 MIN:32 SEC:58, validation accuracy: 0.926**

**Best accurancy - model: new_convolutional_model: validation accuracy: 0.930**

**Total elapsed time: DAYS:0 HOURS:1 MIN:34 SEC:12**


![](https://www.radikal.kz/images/2019/02/17/BEZYMYNNYIbbb05c7422f19d7f.png)

