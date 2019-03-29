# TCR-Epitope Modeling

The goal of this project was to learn more about T cell receptors, epitope binding, and the challenges of building a model capable of taking any given TCR sequence and epitope sequence and predicting their interaction.

This work is heavily inspired by the work of *Jurtz et. al.* [NetTCR: sequence-based prediction of TCR binding to peptide-MHC complexes using convolutional neural networks](https://www.biorxiv.org/content/10.1101/433706v1).

CDR3 alpha, beta, or both were one-hot encoded to serve as input to the first convolution layer. Corresponding epitope sequences were also one-hot encoded, rather than using the raw sequences. This was done to reduce the total number of features introduced to the model. The ultimate goal of building a predictive model of TCR-epitope binding would require the epitope sequence for *de novo* predictions. The limited number of examples in public databases make this an unlikely achievable goal. For this project, I limited the scope to 4 epitopes (NLVPMVATV, GILGFVFTL, GLCTLVAML, LLWNGPMAV) presented by the MHCI allele HLA-A\*02:01.

### CNN Architecture
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/model_architecture.png "CNN architecture")

### Data processing and training
All TCR-epitope sequences corresponding to HLA-A\*02:01 were downloaded from IEDB. There are 6718 unique CDR3α sequences, 10055 unique CDR3β sequences, and only

### Model performance using the complete CDR3 alpha or beta chain
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_or_b_cdr_model_performance.png "Complete alpha or beta models")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_or_b_cdr_f1_prec_rec_acc.png "Complete alpha or beta models")


### Model performance using only the V & J components
In a similar manner to *Jurtz et. al.*, I evaluated the importance of the information in the V & J sequence elements alone. To do so, I masked (replaced with X) the internal part of the sequence and left the two AAs on the left side and the two AAs on the right side. For the joined CDR3α and CDR3β model, each was masked separately and then concatenated such that the middle of the sequence has four AAs (the left and right end both have two AAs).

Comparing these results to the models trained on full sequences should be taken with the following caveat. This process reduced the number of unique sequences in the library. To prevent data leakage (the same sequences in the train and test set), all duplicate CDRs were moved to the training set, so the number of sequences for testing was reduced.

![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/alpha_vj_only.png "VJ-only alpha model")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/beta_vj_only.png "VJ-only beta model")
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_chain_vj_only.png "VJ-only alpha & beta model")


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
