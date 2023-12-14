import warnings
warnings.filterwarnings("ignore")

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from seqeval.metrics import f1_score, classification_report
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import *
from model import TransformerTagger, generate_square_subsequent_mask
from options import get_options
args = get_options()

learning_rate = args.lr
epochs = args.epochs
embedding_dim = args.embedding_dim            
hidden_dim = args.hidden_dim               
nlayers = args.nlayers                   
nhead = args.nhead                      
dropout = args.dropout                 
batch_size = args.batch_size
device = args.device
path = args.path

training_data_path = path + "train.txt"
validation_data_path = path + "valid.txt"
test_data_path = path + "test.txt"
training_data = dataset_build_with_batch(training_data_path, batch_size)
validation_data = dataset_build(validation_data_path)
testing_data = dataset_build(test_data_path)


def validate(epoch, data_, model, word2idx, tag2idx, idx2tag, num_of_batches):
    data = copy.deepcopy(data_)
    tag_ground_truth = []
    tag_prediction = []
    total_loss = 0.0

    for i in range(len(data)):
        tag_ground_truth.append(data[i][1])
        data[i] = ([word2idx[t] for t in data[i][0]], [tag2idx[t] for t in data[i][1]])

    dataset_word = []
    dataset_tag = []
    for data_tuple in data:
        dataset_word.append(data_tuple[0])
        dataset_tag.append(data_tuple[1])

    with torch.no_grad():
        for data_tuple in data:

            # generate mask
            mask = generate_square_subsequent_mask(1).to(device)

            # convert the data to tensor
            inputs = torch.tensor([data_tuple[0]], dtype=torch.long).to(device)
            y = torch.tensor([data_tuple[1]], dtype=torch.long).to(device)

            # validate the model
            tag_scores = model(inputs, mask)
            loss_function = nn.CrossEntropyLoss()
            tag_scores_ = tag_scores.view(-1, tag_scores.shape[2])
            y_ = y.view(y.shape[0] * y.shape[1])
            loss = loss_function(tag_scores_, y_)
            total_loss += loss

            # convert the tag score to predicted tags
            if torch.cuda.is_available():
                tag_scores = tag_scores.cpu().detach()
            pred_raw = torch.argmax(tag_scores, dim=2)
            pred = [idx2tag[str(int(t))] for t in pred_raw[0]]
            tag_prediction.append(pred)

    print("Epoch {} Validation loss: ".format(epoch+1), float(total_loss / num_of_batches))
    print(classification_report(tag_ground_truth, tag_prediction))

    return f1_score(tag_ground_truth, tag_prediction), float(total_loss / num_of_batches)


def train():

    # load word & tag dictionaries
    word2idx = word_to_idx([training_data_path, validation_data_path, test_data_path])
    tag2idx, idx2tag = tag_to_idx(training_data_path)

    # convert the words and tags into indexes
    for i in range(len(training_data)):
        training_data[i] = ([word2idx[t] for t in training_data[i][0]], [tag2idx[t] for t in training_data[i][1]])

    # reformat the dataset
    dataset_word = []
    dataset_tag = []
    for data_tuple in training_data:
        dataset_word.append(data_tuple[0])
        dataset_tag.append(data_tuple[1])

    # convert the data to tensors and load into torch dataset
    torch_set = Data.TensorDataset(torch.tensor(dataset_word, dtype=torch.long), torch.tensor(dataset_tag, dtype=torch.long))
    loader = Data.DataLoader(
        dataset=torch_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # model settings
    model = TransformerTagger(len(word2idx), len(tag2idx), embedding_dim, nhead, hidden_dim, nlayers, dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # record training, validation loss and validation f1 score of each epoch
    y_list = []
    x_list = []
    z_list = []
    a_list = []
    min_valid_loss = float('inf')

    # Training
    for epoch in range(epochs):
        training_loss = 0.0
        with tqdm(loader, desc='Epoch {}'.format(epoch + 1)) as pbar:
            for i, (batch_x, batch_y) in enumerate(pbar):

                # generate mask
                mask = generate_square_subsequent_mask(len(batch_x)).to(device)

                # forward the model
                model.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                tag_scores = model(batch_x, mask)

                # calculate loss and backward
                tag_scores = tag_scores.view(-1, tag_scores.shape[2])
                batch_y = batch_y.view(batch_y.shape[0] * batch_y.shape[1])
                loss = loss_function(tag_scores, batch_y)
                training_loss += loss / len(loader)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                pbar.set_description("Epoch{}, Loss:{:.3}".format(epoch+1, loss))

        # validation
        f1, valid_loss = validate(epoch, validation_data, model, word2idx, tag2idx, idx2tag, len(loader))

        x_list.append(epoch + 1)
        y_list.append(valid_loss)
        a_list.append(f1)
        if torch.cuda.is_available():
            training_loss = training_loss.cpu()
        z_list.append(training_loss.detach())

        # if validation f1 score becomes the current minimum, save the model
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), "Transformer_{}_{}_{}_{}.pth".format(args.batch_size, args.dropout, args.embedding_dim, args.hidden_dim))


    # plot the loss graph
    plt.plot(x_list, y_list, label='Validation loss')
    plt.plot(x_list, z_list, label='Training loss')
    plt.legend(loc="upper left")
    plt.grid(True, axis='y')
    plt.savefig("loss_transformer.png")

    # plot the f1 score graph
    plt.figure()
    plt.plot(x_list, a_list, label='F1 score')
    plt.legend(loc="upper left")
    plt.grid(True, axis='y')
    plt.savefig("f1_transformer.png")

if __name__ == "__main__":
    train()