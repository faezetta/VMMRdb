# MI-CNN-based VMMR

A Multiple Instance Learning Convolutional Neural Networks approach to VMMR, to overcome the constrain of limited amount of well labeled data.<br/>
Instead of feeding the training image into the network and following the global learning approach, here we formulate each image into a bag consisting of multiple instances.<br/>
CNN loss function is formulated for MIL as below:<br/>
<p align="center"><img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/mil_loss.png" alt="CNN loss function" width="200px">
<img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/mil_crossentropy.png" alt="MI-CNN loss function" width="230px">

<p align="center"><img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/milcnnFlowChart.png" alt="MIL-based CNN for VMMR" width="600px">

The code is based on [`fb.resnet.torch`](https://github.com/facebookarchive/fb.resnet.torch) with modification required for MIL. 

```
OMP_NUM_THREADS=1 th main.lua -data DATA_DIR -nThreads 8 -nClasses 3036 -save CHECKPOINT_DIR  -nEpochs 200 -batchSize 1 -LR 0.0001 -netType resnet_pre2 | tee log_VMMRdb_MIL_Resnet.txt
```
