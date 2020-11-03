# MI-CNN-based VMMR

A Multiple Instance Learning Convolutional Neural Networks approach to VMMR, to overcome the constrain of limited amount of well labeled data.<br/>
Instead of feeding the training image into the network and following the global learning approach, here we formulate each image into a bag consisting of multiple instances.<br/>
CNN loss function is formulated for MIL as below:<br/>
<p align="center"><img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/mil_loss.png" alt="CNN loss function" width="200px">
<img align="center" src="https://github.com/faezetta/VMMRdb/blob/master/meta/mil_crossentropy.png" alt="MI-CNN loss function" width="200px">
