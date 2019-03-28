import pandas as pd
import numpy as np
import random

aminos = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aminos_dict = {a:i for i,a in enumerate(aminos)}

def load_and_prepare_data(iedb_csv, chain, select_epitopes, V_and_J_only=False, show_summary=False):
    df= pd.read_csv(iedb_csv, dtype=object)
    df = df.rename(columns={'Description':'epitope', 'Chain 1 CDR3 Curated':'chain1', 'Chain 2 CDR3 Curated':'chain2',
                           'Curated Chain 1 V Gene': 'v1', 'Curated Chain 2 V Gene':'v2'})

    # Remove duplicate alpha and duplicate beta chains (keep first)
    df = df.drop_duplicates(subset=['chain1', 'chain2'])
    df['bind'] = 1
    df['set'] = 'true_binders'

    if chain == 'full_chain':
        df = df.drop_duplicates(subset=['chain1'])
        df = df.drop_duplicates(subset=['chain2'])
        df['full_chain'] = df['chain1'] + df['chain2']
        df['length'] = df['full_chain'].str.len()
        df = df[df['length'].between(20,31)]
        df = df[['full_chain', 'epitope', 'chain1', 'chain2', 'v1', 'v2', 'set', 'bind']]
    else:    
        df = df[['epitope', 'chain1', 'chain2', 'v1', 'v2', 'set', 'bind']]
        df = df[df[chain].str.len() < 20]

    df = df.dropna(subset=[chain])
    df = df.drop_duplicates(subset=[chain])
    df = df[df[chain].str.isalpha()]
    df = df[df[chain].str.len() > 5]
    # Some beta chain sequences from the IEDB database contain non-amino acids. Removing these for now.
    df = df[df[chain].apply(lambda x: ('O' not in x) & ('B' not in x) & ('f' not in x) & ('c' not in x) & ('U' not in x) & ('J' not in x) & ('k' not in x))]

    sub = df[df['epitope'].apply(lambda x: x in select_epitopes)]
    sub.reset_index(inplace=True, drop=True)

    other_ep_tcrs = df[~df['epitope'].isin(sub['epitope'])]
    other_ep_tcrs = other_ep_tcrs.drop_duplicates(subset=[chain])
    other_ep_tcrs.reset_index(inplace=True, drop=True)
    other_ep_tcrs['bind'] = 0
    other_ep_tcrs['set'] = 'other_ep_tcrs'

    if V_and_J_only:
        sub.loc[:, chain] = sub.loc[:, chain].apply(lambda x: reduce_seq_to_V_and_J(x))
        other_ep_tcrs.loc[:, chain] = other_ep_tcrs.loc[:, chain].apply(lambda x: reduce_seq_to_V_and_J(x))

    if show_summary:
        print('TCR-epitope breakdown after filtering:')
        print(sub['epitope'].value_counts())
        print('\nTCR-epitopes not in the main set:')
        print(other_ep_tcrs['epitope'].value_counts())
        
    return sub, other_ep_tcrs

def make_synthetic_data(data, chain, epitopes, synthetics_to_make, tile_sequences=False):
    synthetic_types = {}
    synthetic_types['swapped_epitopes'] = pd.DataFrame({chain:data[chain], 'epitope':data['epitope'].apply(lambda current_ep: swap_epitopes(current_ep, epitopes)),
                                                        'bind':0, 'set':'swapped_epitopes'})
    synthetic_types['scrambled_seqs_and_random_eps'] = pd.DataFrame({chain:data[chain].apply(randomize_seq), 'epitope': [np.random.choice(epitopes) for i in range(len(data))],
                                                                     'bind':0, 'set':'scrambled_seqs_and_random_eps'})
    synthetic_types['scrambled_seqs_and_same_eps'] = pd.DataFrame({chain:data[chain].apply(randomize_seq), 'epitope': data['epitope'],
                                                                   'bind':0, 'set':'scrambled_seqs_and_same_eps'})   
#     synthetic_types['internal_random_buffer1'] = make_internal_random_chain_data(data, chain=chain, num_eps=epitopes.nunique(), buffer=1, set_name='internal_random_buffer1')
#     synthetic_types['internal_random_buffer2'] = make_internal_random_chain_data(data, chain=chain, num_eps=epitopes.nunique(), buffer=2, set_name='internal_random_buffer2')

    synthetic_data = []
    for _type in synthetics_to_make:
        synthetic_data.append(synthetic_types[_type])
    data = pd.concat(synthetic_data)

    if tile_sequences:
        tiled_sequences = []
        tcr_max_len = data[chain].str.len().max()

        for i in range(len(data)):
            tiled_sequences.append(tile_sequence(data.iloc[i:i+1], chain, tcr_max_len))
        data = pd.concat(tiled_sequences, axis=0)
        
    return data

def reduce_seq_to_V_and_J(seq):
    X_len = len(seq) - 4
    return seq[:2] + 'X' * X_len + seq[-2:]

def one_hot_encode(seq):
    arr = np.zeros((len(aminos_dict), len(seq)))
    for i, aa in enumerate(seq):
        if aa == 'X':
            continue
        else:
            arr[aminos_dict[aa], i] = 1
    return arr

def encode_epitopes(epitopes):
    d = {}
    for i, _class in enumerate(epitopes):
        arr = np.zeros(len(epitopes), dtype=int)
        arr[i] = 1
        d[_class] = arr      
    return d

# def encode_epitopes(data, class_col):
#     classes = data[class_col].unique()
#     d = {}
#     for i, _class in enumerate(classes):
#         arr = np.zeros(len(classes), dtype=int)
#         arr[i] = 1
#         d[_class] = arr
        
#     return data[class_col].apply(lambda x: d[x])

def swap_epitopes(current_ep, epitopes):
    new_ep = current_ep
    while new_ep == current_ep:
        new_ep = np.random.choice(epitopes)
    return new_ep

def randomize_seq(seq):
    return ''.join(random.sample(seq,len(seq)))

def give_each_tcr_all_epitopes(data, seq_col, num_eps, bind=0):
    eps = dummy_ep_one_hots(num_eps)
    out = []
    for seq in data[seq_col]:
        for ep in eps:
            out.append([seq, ep, bind])
            
    return pd.DataFrame(out, columns=[seq_col, 'oh_ep', 'bind'])

def dummy_ep_one_hots(num_eps):
    eps = []
    for i in range(num_eps):
        t = np.zeros(num_eps, dtype=int)
        t[i] = 1
        eps.append(t)
    return eps

def randomize_inner_chain(seq, buffer=2):
    try:
        seq = seq.replace(' ', '')
        edit_len = len(seq) - buffer*2
        rand_str = ''.join([np.random.choice(aminos) for x in range(edit_len)])
        new_seq = seq[:2] + rand_str + seq[-2:]
    except:
        return 'no seq'
    return new_seq

def make_internal_random_chain_data(data, chain, chain1_col='chain1', chain2_col='chain2', set_name='internal_rand', num_eps=2, buffer=2, bind=0):
    if chain == 'full_chain':
        chain1 = data['chain1'].apply(randomize_inner_chain)
        chain2 = data['chain2'].apply(randomize_inner_chain)
        data = pd.DataFrame({chain1_col:chain1, chain2_col:chain2, 'full_chain': chain1 + chain2})
    else:
        data[chain] = data[chain].apply(randomize_inner_chain)
    
    data['bind'] = bind
    data['set'] = set_name
    ep_choices = dummy_ep_one_hots(num_eps)
    data['oh_ep'] = [ep_choices[np.random.randint(0, len(ep_choices)-1)] for x in range(len(data))]
    
    return data


def tile_sequence(row, seq_col, target_length):
    seq = row[seq_col].reset_index(drop=True)[0]
    max_pos = target_length - len(seq) + 1
    expanded_df = pd.concat([row] * max_pos, axis=0)
    expanded_df.reset_index(inplace=True, drop=True)
    seqs = []
    for pos in range(max_pos):
        padded = 'X'*pos + seq + 'X'*(target_length - len(seq) - pos)
        seqs.append(padded)
        
    expanded_df.loc[0:, seq_col] = seqs
    
    return expanded_df

def center_and_pad_sequence(seq, target_length):
    pos = int(np.ceil(((len(seq) + target_length) / 2) - len(seq)))
    padded = 'X'*pos + seq + 'X'*(target_length - len(seq) - pos)
    return padded

def encode_seqs(seqs, target_length):
    encoded = []
    
    for seq in seqs:
        if len(seq) < target_length:
            seq = center_and_pad_sequence(seq, target_length)
        encoded.append(one_hot_encode(seq))
    return np.array(encoded)

def subtract_from_ends(seq, left_end, right_end):
    return seq[left_end:len(seq) - right_end]