# TCR-Epitope Modeling


### CNN Architecture
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/model_architecture.png "CNN architecture")

> Conv. Layer 1:
> 	Filters:70
> 	Kernel size: 8
> 	Activation: ReLU
> Max pooling: 4
>
> Conv. Layer 2:
>	Filters:70
>	Kernel size: 8
>	Activation: ReLU
>Max pooling: 4
>
>Dense (Epitope input):
>	Nodes: number of unique epitopes in training and test set
>	Activation: Linear
>	Concatenated with 2nd max pooling
>
>Dense:
>	Nodes 200
>	Dropout probability: 0.2
>	Activation: ReLU

### Model performance using the complete CDR3 alpha or beta chain
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_or_b_cdr_model_performance.png "Complete alpha or beta models")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_or_b_cdr_f1_prec_rec_acc.png "Complete alpha or beta models")


### Model performance using only the V & J components
#### Only the last two amino acids from the left and right side are retained. Then internal sequence was replaced with X's.
A la *Jurtz et. al.*
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/vj_only_ALPHA_cdr_model_performance.png "VJ-only alpha model")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/vj_only_BETA_cdr_model_performance.png "VJ-only beta model")

