
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, Multiply, MaxPooling1D, Concatenate
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
from keras.utils import plot_model
import scipy.stats as stats

def train_test_split(data, classes, class_col, chain, train_size=0.75, seed=111, put_dups_in_train=False):
    train = []
    test = []  

    if put_dups_in_train:
        if chain == 'full_chain':
            dups = data[(data['chain1'].duplicated()) | (data['chain2'].duplicated())]
            data = data[~((data['chain1'].duplicated()) | (data['chain2'].duplicated()))]
            train.append(dups)
        else:
            dups = data[data[chain].duplicated()]
            data = data[~data[chain].isin(dups[chain])]
            train.append(dups)
    
    for _class in classes:
        t = data[data[class_col] == _class]
        t = t.sample(frac=1)
        
        split = int(np.ceil(len(t) * train_size))
        
        train.append(t[:split])
        test.append(t[split:])
        
    return pd.concat(train).reset_index(drop=True), pd.concat(test).reset_index(drop=True)


def eval_predictions(truth, pred_binary):
    f1 = f1_score(truth, pred_binary)
    prec = precision_score(truth, pred_binary)
    rec = recall_score(truth, pred_binary)
    acc = accuracy_score(truth, pred_binary)
    print('F1 = {0:.3f}    precision = {1:.3f}    recall = {2:.3f}    accuracy = {3:.3f}'.format(f1, prec, rec, acc))
    return f1, prec, rec, acc

def build_model(train_tcrs, train_eps, use_one_hot=True):
    if use_one_hot:
        seq_len = train_tcrs[0].shape[1]
        input_tcr = Input(shape=(21,seq_len,))
    else:
        seq_len = train_tcrs.shape[1]
        input_tcr = Input(shape=(seq_len,21,))

    ep_len = train_eps.shape[1]
    input_ep = Input(shape=(ep_len,))

    model = Model()

    x = Conv1D(activation="relu", padding='same', filters=70, kernel_size=8)(input_tcr)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(activation="relu", padding='same', filters=70, kernel_size=8)(x)
    x = MaxPooling1D(pool_size=4)(x)


    flattened = Flatten()(x)

    input2 = Dense(ep_len, activation='linear')(input_ep)
    x = Concatenate()([flattened, input2])


    x = Dense(100, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_tcr, input_ep], outputs=x)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #adam = keras.optimizers.Adam()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0.002, restore_best_weights=True)]
    
    return model, callbacks

class BatchModelResults:
    def __init__(self):
        self.saved_test_sets = []
        self.f1_scores = []
        self.prec_scores = []
        self.rec_scores = []
        self.acc_scores = []
        
    def add_test_output(self, test_data, truth_col='bind', pred_col='pred'):
        truth = test_data[truth_col]
        pred = test_data[pred_col]
        
        self.saved_test_sets.append(test_data)
        self.f1_scores.append(f1_score(truth, pred))
        self.prec_scores.append(precision_score(truth, pred))
        self.rec_scores.append(recall_score(truth, pred))
        self.acc_scores.append(accuracy_score(truth, pred))

    def plot_all_auroc(self, truth_col='bind', pred_col='prob'):
        f, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        aucs = []
        
        for test_set in self.saved_test_sets:
            truth = test_set[truth_col]
            pred = test_set[pred_col]
            fpr, tpr, _ = roc_curve(truth, pred)
            aucs.append(auc(fpr, tpr))
            ax.plot(fpr, tpr, color='darkorange', alpha=0.5, lw=3)
    
        ax.set_xlim(-0.01,1)
        ax.set_ylim(0,1.01)
        ax.set_xlabel('FPR', size=14)
        ax.set_ylabel('TPR', size=14)
        ax.set_title('Mean AUC: {0:.2f}'.format(np.mean(aucs)))
        
    def plot_set_accuracy(self):
        sets = np.sort(self.saved_test_sets[0]['set'].unique())

        f, axes = plt.subplots(ncols=len(sets), nrows=1, sharey=True)
        f.set_size_inches(len(sets) * 3, 4)

        for i, _set in enumerate(sets):
            num_pred_bind = []
            num_pred_non_bind = []
            accuracies = []
            for test_pred in self.saved_test_sets:
                set_data = test_pred[test_pred['set'] == _set]

                expected_bind = set_data[(set_data['bind'] == 1)].shape[0]
                expected_non_bind = set_data[(set_data['bind'] == 0)].shape[0]

                predicted_bind = set_data[(set_data['pred'] == 1.0)].shape[0]
                num_pred_bind.append(predicted_bind)

                predicted_non_bind = set_data[(set_data['pred'] == 0)].shape[0]
                num_pred_non_bind.append(predicted_non_bind)

                acc = (set_data['bind'] == set_data['pred']).sum() / len(set_data)
                accuracies.append(acc)

            bind_std_err = stats.sem(num_pred_bind)
            non_bind_std_err = stats.sem(num_pred_non_bind)

            bind_mean = np.mean(num_pred_bind)
            non_bind_mean = np.mean(num_pred_non_bind)


            ax = axes[i]
            ax.bar([1,2], height=[expected_bind, expected_non_bind], alpha=0.5, label='Expected', linewidth=2, edgecolor='k', color='w')
            ax.bar([1,2], height=[bind_mean, non_bind_mean], alpha=0.5, label='Predicted', color='darkorange', yerr=[bind_std_err, non_bind_std_err])
            ax.set_xticks(range(1,3))
            ax.set_xticklabels(['Bind', 'No Bind'])
            ax.set_title('{0}\nMean Acc: {1:.3f}'.format(_set, np.mean(accuracies)))

        ax.legend()
        
    def plot_metrics_boxplots(self):
        f, ax = plt.subplots()
        ax.boxplot([self.f1_scores, self.prec_scores, self.rec_scores, self.acc_scores], labels=['F1', 'Precision', 'Recall', 'Accuracy'])
        ax.set_ylim(0,1.01)
        ax.set_title('Overall Performance')

    def plot_epitope_performance(self):
        epitopes = np.sort(self.saved_test_sets[0]['epitope'].unique())

        f, axes = plt.subplots(ncols=len(epitopes), sharey=True)
        f.set_size_inches(len(epitopes)*4, 4)

        for i, ep in enumerate(epitopes):
            ax = axes[i]
            f1_scores = []; prec_scores = []; rec_scores = []; acc_scores = []

            for test_set in self.saved_test_sets:
                sub = test_set[test_set['epitope'] == ep]
                f1_scores.append(f1_score(sub['bind'], sub['pred']))
                prec_scores.append(precision_score(sub['bind'], sub['pred']))
                rec_scores.append(recall_score(sub['bind'], sub['pred']))
                acc_scores.append(accuracy_score(sub['bind'], sub['pred']))
            
            ax.set_ylim(0,1.01)
            ax.boxplot([f1_scores, prec_scores, rec_scores, acc_scores], labels=['F1', 'Prec.', 'Rec.', 'Acc.'])
            ax.set_title('{} ({} samples in test)'.format(ep, len(sub)))