# TCR-Epitope Modeling

The goal of this project was to learn more about T cell receptors, epitope binding, and the challenges of building a model capable of taking any given TCR sequence and epitope sequence and predicting their interaction.

This work is heavily inspired by the work of *Jurtz et. al.* [NetTCR: sequence-based prediction of TCR binding to peptide-MHC complexes using convolutional neural networks](https://www.biorxiv.org/content/10.1101/433706v1). There are several differences between their work and this project. This includes the CNN architecture and the CDR3 sequence and epitope sequence encoding scheme. I also expanded the analysis to include CDR3α, CDR3β, and the concatenation of the two.

CDR3 alpha, beta, or both were one-hot encoded to serve as input to the first convolution layer. The sequences of the epitopes were not used. Instead they were encoded as categorical variables. This was done to reduce the total number of features introduced to the model. The ultimate goal of building a predictive model of TCR-epitope binding would require the epitope sequence for *de novo* predictions. The limited number of examples in public databases make this a difficult goal. For this project, I limited the scope to 4 epitopes (NLVPMVATV, GILGFVFTL, GLCTLVAML, LLWNGPMAV) presented by the MHCI allele HLA-A\*02:01.

### CNN Architecture
See hyperparameters below.
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/model_architecture.png "CNN architecture")

### Data processing and training
All TCR-epitope sequences corresponding to HLA-A\*02:01 were downloaded from IEDB. There are 6718 unique CDR3α sequences, 10055 unique CDR3β sequences, and only 890 unique CDR3α-CDR3β pairs. Considering that these are all positive examples of binding, synthetic non-binders had to be made (*Jurtz el. al.*). This was done in two ways. The first involved taking each CDR3 sequence and assigning it an epitope other than the one it is known to bind to. For the second, the CDR3-epitope pairs were kept the same but the CDR3 sequence was shuffled. The other CDR3-epitope pairs that did not correspond to the four epitopes in the IEDB data were included as non-binders and were randomly assigned one of the four.

When making train / test splits, it was ensured that the train set contained 70% of each epitope and the test set 30% of each epitope. Seven models with random splits of the data were evaluated. Models were trained for however many epochs it took before validation (10% of training data) loss bottomed out (usually around 7). Training on scrambled data produced models that were not capable of predicting binding (AUROCs ~ 0.5).

## Model performance on predicting CDR3-epitope binding using CDR3α, CDR3β, or both chains
Training on each sequence set showed similar discriminative ability. It would make sense that having both CDR3α and CDR3β would provide more information that would increase the accuracy of the model and the results here suggest that that could be the case. However, it's possible that the sequences in the paired set are more similar than those that are not paired or that there are other differences relating to the source of the data that could falsely inflate the models generalizability. All models struggled to correctly classify the 'other epitopes' CDR3s, which is a significant weakness. The paired CDR3α and CDR3β model was the most discriminative with these 'other'  CDR3-epitope pairs.

![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_b_or_both_model_performance.png "Complete alpha, beta, or both models")

### Evaluation on each of the 4 epitopes
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_a_b_or_both_f1_prec_rec_acc.png "Complete alpha, beta, or both models")


## Model performance using only the V & J components
In a similar manner to *Jurtz et. al.*, I evaluated the importance of the information in the V & J sequence elements alone. To do so, I masked (replaced with X) the internal part of the sequence and left the two AAs on the left side and the two AAs on the right side. When one-hot encoding, an 'X' is converted to an array of all 0s. For the joined CDR3α and CDR3β model, each was masked separately and then concatenated such that the middle of the sequence has four AAs and the left and right end both have two AAs. (ex. VCXXXXXRALAXXXXPL)

Comparing these results to the models trained on full sequences should be taken with the following caveat. This process reduced the number of unique sequences in the library. To prevent data leakage (the same sequences in the train and test set), all duplicate CDR3s were moved to the training set, so the number of sequences for testing was reduced.

Ultimately, I think it's safe to say that the V & J segments are contributing significantly more to model performance than the internal sequence. This matches other reports: Jurtz et. al., Dash et. al., Glanville et. al.

### CDR3α VJ-only
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/alpha_vj_only.png "VJ-only alpha model")

### CDR3β VJ-only
The ability of a CDR3β VJ-only model to discriminate between binders and non-binders is significantly reduced in comparison to full sequence models.
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/beta_vj_only.png "VJ-only beta model")

### CDR3α & CDR3β VJ-only
![alt text](https://github.com/pjsample/TCR-Epitope_Modeling/blob/master/images/full_chain_vj_only.png "VJ-only alpha & beta model")


### Model hyperparameters
The best hyperparameters and model architecture were not systematically determined. Instead I used a similar setup as what I used for my human 5'UTR research project: [Human 5′ UTR design and variant effect prediction from a massively parallel translation assay](https://www.biorxiv.org/content/10.1101/310375v1). I played around with different kernel sizes, max pooling sizes, nodes in the dense layer, and drop out. For the most part, kernel sizes between 4 - 8, max pooling from 2 - 4, and dense layer nodes between 100 - 200 worked well.

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
