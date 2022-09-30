import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tensorflow import keras
from typing import List, Tuple, Dict, Union
from collections import Counter

vocab = [char for char in 'ACDEFGHIKLMNPQRSTVWY'] + ['$']
one_hot = F.one_hot(torch.arange(0, len(vocab)))
encode_dict = dict(zip(vocab, one_hot))
decode_dict = dict(zip([i for i in range(len(vocab))], vocab))
freqList: List[Tuple[str, int]]

def decodeOneHot(oneHotSeq: torch.Tensor) -> str:
    '''
    Decode sequence of one-hot vectors back to the original string
        Args: 
            * oneHotSeq: tensor of shape (1,100,21)
        Returns:
            * original protein sequence string
    '''
    if oneHotSeq.shape[0] != 1:
        raise ValueError('Function only defined for decoding one sequence at a time.')

    return ''.join([decode_dict[x.nonzero().squeeze().item()] for x in oneHotSeq[0,:-1,:] if torch.sum(x) > 0])
    

def sample(predictions, temperature: Union[float, List[float]] = 1.18) -> np.int64:
    '''
    Sample from predictions with increased diversity. 
    increased temps -> increased stochasticity
        Args: 
            * predictions: predictions for each character in the vocab 
                - shape: (20,)
        Returns:
            * index of prediction after sampling with temperature     
    '''
    if isinstance(temperature, list):
        ...
    else:
        preds = np.asarray(predictions[:-1]).astype('float64')
        with np.errstate(divide='ignore'):
            preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        return np.argmax(probs)
    
def diversityScore(seq: str) -> np.float64:
    '''
    Test diversity of a single sequence using Shannon diversity
        Args: 
            * seq:  string of amino acids
        Returns:
            * Shannon diversity of seq as float64
    '''
    freq = list(Counter(list(seq)).values())
    seqLen = len(seq)
    alphSize = len(vocab) - 1
    
    charCounts = np.array(freq)
    ent = -np.sum((charCounts/seqLen) * np.log(charCounts/seqLen))

    return np.round(abs(ent / np.log(alphSize)), decimals=5)

def deNovoScore(seqSet) -> np.float64:
    '''
    todo     
    '''
    ...

def plotFrequencies(freq: np.ndarray, startFreq: np.ndarray, ngramList: np.ndarray) -> None:
    ''' 
    Plot figures for:
        - general amino acid frequencies
        - starting amino acid frequencies
        - bigram, trigram, and quadgram frequencies (top 10 of each)

        Args:
            * freq:         np.array of all amino acid occurences in the dataset
            * startFreq:    np.array of all starting amino acid occurences in the dataset   
            * ngramList:    np.array of all bigram, trigram, and quadgram occurences in the dataset
        Return:
            * None. All figures are saved as 'png' files in the 'figures' folder in the cwd
    '''
    sns.histplot(freq, palette='deep')
    plt.savefig('figures/character_frequency.png')
    plt.clf()
    sns.histplot(startFreq, palette='deep')
    plt.savefig('figures/start_character_frequency.png')
    plt.clf()

    ngramList = sorted(ngramList.tolist(), key=len)
    bigramList = [x for x in ngramList if len(x) == 2]
    trigramList = [x for x in ngramList if len(x) == 3]
    quadgramList = [x for x in ngramList if len(x) == 4]
    
    counts = Counter(bigramList)
    bigrams = pd.DataFrame({k:list(v) for k,v in list(zip(['Bigrams', 'Counts'], zip(*counts.items())))})

    counts = Counter(trigramList)
    trigrams = pd.DataFrame({k:list(v) for k,v in list(zip(['Trigrams', 'Counts'], zip(*counts.items())))})

    counts = Counter(quadgramList)
    quadgrams = pd.DataFrame({k:list(v) for k,v in list(zip(['Quadgrams', 'Counts'], zip(*counts.items())))})

    sns.barplot(x='Counts', y='Bigrams', data=bigrams[:10])
    plt.savefig('figures/bigram_frequency.png')
    plt.clf()
    
    sns.barplot(x='Counts', y='Trigrams', data=trigrams[:10])
    plt.savefig('figures/trigram_frequency.png')
    plt.clf()
    
    sns.barplot(x='Counts', y='Quadgrams', data=quadgrams[:10])
    plt.savefig('figures/quadgram_frequency.png')
    plt.clf()


def buildDatasets(args, file='data/pdb_seqres.txt'):
    ''' 
    Build train/test datasets using the data in the 'data' folder of the cwd
    Calculate diversity scores for dataset (todo: move out of function)
    Processes frequencies for plotting (todo: move out of function)
        
        Args:
            * args:             (argsparse.args) only needed for plotting frequencies
            * file:             path for dataset relative to cwd
        Returns:
            Tuple of --
            * diversity:        a dictionary of diversity scores for dataset
                                (keys: 'train', 'test')
            * seq_train         tensor of one-hot encoded protein sequences for training
            * train_labels      corresponding tensor of labels for training
            * seq_test          tensor of one-hot encoded protein sequences for testing (every 10th)
            * test_labels       corresponding tensor of labels for testing
    '''
    prechkd_path = os.path.join(os.getcwd(), file)
    path = prechkd_path if os.path.exists(prechkd_path) else None

    if path is None:
        print('invalid path in preprocess')
        exit()

    with open(file, 'r') as f:
        seq = f.readlines()
        seq = [x.replace('\n', '') for x in seq]

    #bound sequence length to 2 <= n <= 100
    truncd = [seq[i] for i in range(len(seq)) if len(seq[i]) <= 100 and len(seq[i]) > 1]

    diversity = {}
    diversity['train'] = np.array([diversityScore(seq) for i,seq in enumerate(truncd) if i % 10 != 0 ])
    diversity['test'] = np.array([diversityScore(seq) for i,seq in enumerate(truncd) if i % 10 == 0 ])
    
    
    countDict = {x: 0 for x in vocab[:-1]}
    startDict = {x: 0 for x in vocab[:-1]}
    freqList = []
    startFList = []
    for seq in truncd:
        for i,c in enumerate(seq):
            if i == 0:
                startFList.append(c)
                startDict[c] += 1
            freqList.append(c)
            countDict[c] += 1

    if args.plot_freq:
        ngramList = []
        for seq in truncd:
            i = 0
            while i < len(seq):
                if i > 1:
                    if i < len(seq)-4:
                        for s in range(2,5):
                            ngramList.append(seq[i:i+s])
                        i += 4
                        continue
                    if i < len(seq)-3:
                        for s in range(2,4):
                            ngramList.append(seq[i:i+s])
                        i += 3
                        continue
                    if i < len(seq)-2:
                        ngramList.append(seq[i:i+2])
                        i += 2
                        continue
                i += 1

        plotFrequencies(np.array(freqList), np.array(startFList), np.array(ngramList))
        exit()

    dataset = [torch.stack([encode_dict[x_i] for x_i in x[:-1]]) for x in truncd]
    labels = torch.stack([torch.stack([encode_dict[lbl].nonzero().squeeze().float() for lbl in seq[-1]]) for seq in truncd])
    padded = torch.nn.utils.rnn.pad_sequence(dataset, batch_first=True)

    padded_data = torch.zeros( (lambda x,y,z: (x,y+1,z))(*padded.shape) )
    padded_data[:,:-1,:] = padded.detach().clone()
    padded_data[:,-1,:] = encode_dict['$']
    
    seq_train = torch.stack([padded_data[i,:,:] for i in range(padded_data.shape[0]) if i % 10 != 0])
    train_labels = torch.stack([labels[i,:] for i in range(labels.shape[0]) if i % 10 != 0])

    seq_test = torch.stack([padded_data[i,:,:] for i in range(padded_data.shape[0]) if i % 10 == 0])
    test_labels = torch.stack([labels[i,:] for i in range(labels.shape[0]) if i % 10 == 0])
    #print(list(x.shape for x in (seq_train, seq_test, train_labels, test_labels)))
    
    freqList = sorted(startDict.items(), key=lambda x: x[1], reverse=True)
    return (diversity,
        seq_train.detach().clone(), 
        train_labels.detach().clone(), 
        seq_test.detach().clone(), 
        test_labels.detach().clone())


def generateSequence(model: keras.Sequential, 
                startAA: Union[str, None], 
                sampTemp: float,
                length: int,
                n: int = 1) -> Union[List[str], str]:
    '''  
    todo     
    '''

    if startAA is None:
        startAA = torch.randint(len(vocab)-1, (1,)).item()
    elif len(startAA) > 1 or startAA not in decode_dict:
        raise ValueError('The starting amino acid must be a single character in the vocabulary.')
    
    res = []      
    for i in range(n):
        start = encode_dict[decode_dict[startAA]]
        empty = torch.zeros(1,100,21)
        empty[0,-1,:] = encode_dict['$'].float()
        for j in range(length):
            next_char = start if j == 0 else encode_dict[decode_dict[pred_idx]]
            empty[0,j,:] = next_char.float()
            
            prediction = model.predict(x=empty.numpy(), batch_size=1).flatten()
            pred_idx = sample(prediction, sampTemp)

        res.append(decodeOneHot(empty))

    return res.pop() if len(res) == 1 else res


#outer dict key: Dict['str(temperature)']
#dict keys within list: List[ D1['seq', 'len', 'div_score'], ..., Dn['seq', 'len', 'div_score'] ] 
sequenceDict = Dict[str,List[Dict[str, Union[str, int, np.float64]]]]
def testGeneration(model: keras.Sequential, temps: List[float] = [0.8, 1.0, 1.2, 1.5], n: int = 5) -> sequenceDict:
    '''
    todo       
    '''
    top5 = [encode_dict[k] for k,_ in freqList[:5]]
    btm5 = [encode_dict[k] for k,_ in freqList[-5:]]
    L = [top5, btm5]
    seqDict = {str(k):[] for k in temps}

    lengths = torch.randint(2, 99, (n,))
    for s in range(len(L)):
        print('%s 5 Most Frequent Starting AAs:' % ('Top' if s == 0 else 'Bottom'))
        for i in range(len(L[s])):
            for j in lengths:
                for k in seqDict.keys():
                    gen = generateSequence(model, L[s][i], float(k), j)
                    print('(t: %2f)\t%s\t(ln: %d)' % (float(k), gen, j))
                    seqDict[k].append({'seq': gen,'len': j, 'div_score': diversityScore(gen)})
                    
    return seqDict


def plotGenerated(seqDict: sequenceDict) -> None:
    '''
    todo       
    '''
    ...

def dependencyTest(model: keras.Sequential, seq_test: torch.Tensor, test_labels: torch.Tensor):
    ''' 
    todo      
    '''
    prediction = model.predict(x=seq_test.numpy(), batch_size=1)

    #
    true_labels = np.asarray(test_labels.flatten(), dtype=int)
    predictions = prediction.argmax(axis=1)

    #
    mod_seq_test = torch.stack([seq_test[i,:,:] for i,correct in enumerate(true_labels == predictions) if correct])
    mod_test_labels = torch.stack([test_labels[i,:] for i,correct in enumerate(true_labels == predictions) if correct])

    #
    rand_idx = torch.randint(mod_seq_test.shape[0], (5,))
    seq_test = mod_seq_test[rand_idx]
    test_labels = mod_test_labels[rand_idx]

    x = test_labels[0].item()

    while x == test_labels[0].item():
        x = torch.randint(len(vocab)-1, (1,)).item()

    last_idx = sum([1 for i in range(seq_test[0].shape[0]) if torch.all(seq_test[0,i,:] == torch.zeros(len(vocab)))])
    altered_seq = torch.zeros(last_idx, *seq_test[0].shape)
    altered_seq[:,:,:] = seq_test[0,:,:]

    true_predictions = model.predict(x=seq_test[0,:,:].unsqueeze(0).numpy(), batch_size=1).flatten()
    true_top3_idx = true_predictions.argsort()[-1:-4:-1]
    true_top3_probs = ['%.10f' % x for x in true_predictions[true_top3_idx]]

    print(true_predictions)
    print(true_top3_idx)
    print(true_top3_probs)
    input()

    str_rep = ''.join([decode_dict[x.nonzero().squeeze().item()] for x in seq_test[0,:last_idx+1,:] if torch.sum(x) > 0])

    for i in range(last_idx):
        altered_seq[i,i,:] = encode_dict[decode_dict[x]]

        print('idx: ', i)
        print('Original: ', str_rep)
        print('Altered: ', str_rep[:i] + '>' + decode_dict[x] + '<' + str_rep[i+1:], end='\n\n')
        prediction = model.predict(x=altered_seq[i,:,:].unsqueeze(0).numpy(), batch_size=1).flatten()
        top3_idx = prediction.argsort()[-1:-4:-1]
        top3_probs = ['%.10f' % x for x in prediction[top3_idx]]

        print('True Top 3:', true_top3_probs)
        print('New Top 3:', top3_probs)
        print('Difference of Top 3: ', ['%.4f' % abs(float(x)-float(y)) for x,y in zip(true_top3_probs, top3_probs)], end='\n\n\n')
        input()

def plotDependencies() -> None:
    '''
    todo
    '''
    ...