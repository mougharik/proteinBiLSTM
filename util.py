import os
import numpy as np
import torch
import torch.nn.functional as F
from tensorflow import keras

vocab = [char for char in 'ACDEFGHIKLMNPQRSTVWY'] + ['$']
one_hot = F.one_hot(torch.arange(0, len(vocab)))
encode_dict = dict(zip(vocab, one_hot))
decode_dict = dict(zip([i for i in range(len(vocab))], vocab))

def buildDatasets(file='data/pdb_seqres.txt'):
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
    
    return seq_train.detach().clone(), train_labels.detach().clone(), seq_test.detach().clone(), test_labels.detach().clone()

def generateSequence(model:keras.Sequential, test_set=None):
    if test_set is None:
        lengths = torch.randint(99, (5,))
        for i in lengths:
            start = encode_dict[decode_dict[torch.randint(len(vocab)-1, (1,)).item()]]
            empty = torch.zeros(1,100,21)
            empty[0,-1,:] = encode_dict['$'].float()
            for j in range(i):
                next_char = start if j == 0 else encode_dict[decode_dict[pred_idx]]
                empty[0,j,:] = next_char.float()
                
                prediction = model.predict(x=empty.numpy(), batch_size=1).flatten()
                pred_idx = prediction.argsort()[-1]

            print('generated protein:', ''.join([decode_dict[x.nonzero().squeeze().item()] for x in empty[0,:-1,:] if torch.sum(x) > 0]))

def dependencyTest(model: keras.Sequential, seq_test: torch.Tensor, test_labels: torch.Tensor):
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
