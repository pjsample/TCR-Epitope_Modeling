# TCR-Epitope Modeling


### CNN Architecture
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/model_architecture.png "CNN architecture")

Conv. Layer 1:
- Filters:70
- Kernel size: 8

Max pooling: 4

Conv. Layer 2:
- Filters:70
- Kernel size: 8

Max pooling: 4

Dense (Epitope input):
- Nodes: number of unique epitopes in training and test set
- Concatenated with 2nd max pooling

Dense:
- Nodes 200
- Dropout probability: 0.2
