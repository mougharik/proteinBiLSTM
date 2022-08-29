
import math
import argparse
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from util import *
from tensorflow.keras import layers



vocab = [char for char in 'ACDEFGHIKLMNPQRSTVWY'] + ['$']
one_hot = F.one_hot(torch.arange(0, len(vocab)))
encode_dict = dict(zip(vocab, one_hot))
decode_dict = dict(zip([i for i in range(len(vocab))], vocab))

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(1)[1]
        loss = F.nll_loss(output, target)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    acc = 100. * correct / total
    avg_loss = sum(losses) / len(train_loader)

    print(acc)
    print(avg_loss)
    return acc, avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * correct / len(test_loader.dataset)), test_loss


def main():
    parser = argparse.ArgumentParser(description='BiLSTM for Protein Sequencing')
    parser.add_argument('--train-model', action='store_true', default=False,
                        help='For loading the \'protein_bilstm\' Model')
    parser.add_argument('--gen', action='store_true', default=True,
                        help='For loading the \'protein_bilstm\' Model')
    parser.add_argument('--deptest', action='store_true', default=True,
                        help='For loading the \'protein_bilstm\' Model')
    
    args = parser.parse_args()
    seq_train, train_labels, seq_test, test_labels = buildDatasets()

    if args.train_model:
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(100,21)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(21, activation='softmax'),
        ])

        model.summary()
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.optimizers.SGD(learning_rate=3e-2),
            metrics=["accuracy"])    
        historyMod = model.fit(x=seq_train.numpy(),
            y=train_labels.numpy(), 
            epochs=10, 
            batch_size=128, 
            validation_data=(seq_test.numpy(), test_labels.numpy()),
            shuffle=True)
        
        writer = SummaryWriter('runs/train')
        writer2 = SummaryWriter('runs/test')

        for x,(y1,y2) in enumerate(zip(historyMod.history['accuracy'], historyMod.history['val_accuracy'])):
            writer.add_scalar('acc', y1, x)
            writer2.add_scalar('acc', y2, x)
        
        for x,(y1,y2) in enumerate(zip(historyMod.history['loss'], historyMod.history['val_loss'])):
            writer.add_scalar('loss', y1, x)
            writer2.add_scalar('loss', y2, x)

        for x,(y1,y2) in enumerate(zip(historyMod.history['loss'], historyMod.history['val_loss'])):
            writer.add_scalar('perp', math.exp(y1), x)
            writer2.add_scalar('perp', math.exp(y2), x)

        model.save('protein_bilstm')

    else:
        model: keras.Sequential = keras.models.load_model('protein_bilstm')
        if args.gen:
            generateSequence(model)
        if args.deptest:
            dependencyTest(model, seq_test, test_labels)

if __name__ == '__main__':
    main()
