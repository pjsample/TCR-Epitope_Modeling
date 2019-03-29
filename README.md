# TCR-Epitope Modeling

The goal of this project was to learn more about T cell receptors, epitope binding, and the challenges of building a model capable of taking any given TCR sequence and epitope sequence and predicting their interaction.

This work is heavily inspired by the work of *Jurtz et. al.* [NetTCR: sequence-based prediction of TCR binding to peptide-MHC complexes using convolutional neural networks](https://www.biorxiv.org/content/10.1101/433706v1).

CDR3 alpha, beta, or both were one-hot encoded to serve as input to the first convolution layer. Corresponding epitope sequences were also one-hot encoded, rather than using the raw sequences. This was done to reduce the total number of features introduced to the model. The ultimate goal of building a predictive model of TCR-epitope binding would require the epitope sequence for *de novo* predictions. The limited number of examples in public databases make this an unlikely achievable goal. For this project, I limited the scope to 4 epitopes (NLVPMVATV, GILGFVFTL, GLCTLVAML, LLWNGPMAV) presented by the MHCI allele HLA-A\*02:01.

### CNN Architecture
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/model_architecture.png "CNN architecture")


### Model performance using the complete CDR3 alpha or beta chain
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_or_b_cdr_model_performance.png "Complete alpha or beta models")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_or_b_cdr_f1_prec_rec_acc.png "Complete alpha or beta models")


### Model performance using only the V & J components
#### Only the last two amino acids from the left and right side are retained. Then internal sequence was replaced with X's.
A la *Jurtz et. al.*
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/vj_only_ALPHA_cdr_model_performance.png "VJ-only alpha model")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/vj_only_BETA_cdr_model_performance.png "VJ-only beta model")

### Model hyperparameters
**Conv. Layer 1:**  
- Filters:70
- Kernel size: 8
- Activation: ReLU  

**Max pooling:** 4

**Conv. Layer 2:**  
- Filters:70
- Kernel size: 8
- Activation: ReLU  

**Max pooling:** 4

**Dense (Epitope input):**  
- Nodes: number of unique epitopes in training and test set
- Activation: Linear
- Concatenated with 2nd max pooling  

**Dense:**  
- Nodes: 200
- Dropout probability: 0.2
- Activation: ReLU  

**Dense:**  
- Nodes: 1
- Activation: sigmoid
- *Predict binding*

### References
[NetTCR: sequence-based prediction of TCR binding to peptide-MHC complexes using convolutional neural networks](https://www.biorxiv.org/content/10.1101/433706v1)
