
import math
import argparse
import random
import tensorflow as tf
from tensorflow import initializers
from torch.utils.tensorboard import SummaryWriter
from utils import *
from keras import Sequential, layers
from keras import regularizers as reg


def buildModel(l2) -> Sequential:
    '''
    todo      
    '''
    wInit = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    bInit = initializers.Constant(1e-1)
    model = Sequential([
        layers.Bidirectional(
                layers.LSTM(
                    units=64, 
                    dropout=0.1,
                    recurrent_dropout=0.1,
                    activation='elu',
                    #recurrent_activation='elu',
                    bias_initializer=bInit,
                    kernel_initializer=wInit,
                    kernel_regularizer=l2, 
                    return_sequences=True), 
                input_shape=(100,21)),
            layers.BatchNormalization(),
            layers.Bidirectional(
                layers.LSTM(
                    units=64, 
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    activation='elu',
                    #recurrent_activation='elu',
                    bias_initializer=bInit,
                    kernel_initializer=wInit,
                    kernel_regularizer=l2, 
                    return_sequences=True)),
            layers.Flatten(),
            layers.Dense(
                    units=256, 
                    activation='relu',
                    kernel_regularizer=l2),
            #layers.Dropout(0.2),
            layers.Dense(units=21, activation='softmax'),
    ])

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description='BiLSTM for Protein Sequencing')
    parser.add_argument('--train-model', action='store_true', default=False,
                        help='For loading the \'protein_bilstm\' Model')
    parser.add_argument('--l2', action='store_true', default=False,
                        help='l2 regularizer')
    parser.add_argument('--plot-freq', action='store_true', default=False,
                        help='plot various frequencies')
    parser.add_argument('--plot-div', action='store_true', default=False,
                        help='plot diversity scores of train/test data')
    parser.add_argument('--gen', action='store_true', default=False,
                        help='generate novel sequences')
    parser.add_argument('--deptest', action='store_true', default=False,
                        help='test long-distance dependencies')
    
    args = parser.parse_args()
    diversity, seq_train, train_labels, seq_test, test_labels = buildDatasets(args)

    if args.train_model:
        if args.l2:
            l2 = reg.l2(1e-3)
        else:
            l2 = None

        model = buildModel(l2)
        model.summary()

        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8, decay=0.0),
            #optimizer=tf.optimizers.SGD(learning_rate=1e-2),
            metrics=["accuracy"])

        historyMod = model.fit(x=seq_train.numpy(),
            y=train_labels.numpy(), 
            epochs=25, 
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
            seqDict = testGeneration(model)
            plotGenerated(seqDict)
        if args.deptest:
            dependencyTest(model, seq_test, test_labels)
            #plotDependencies()

if __name__ == '__main__':
    main()
