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

from toolbox import *
from transformer import TransformerTagger, attention_mask
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
training_data = build_batched(training_data_path, batch_size)
validation_data = build_corpus(validation_data_path)
testing_data = build_corpus(test_data_path)


def transformer_eval(epoch, data_, model, word2idx, tag2idx, idx2tag, num_of_batches):
    data = copy.deepcopy(data_)
    tag_ground_truth, tag_prediction = [], []
    total_loss = 0.0

    for i in range(len(data)):
        tag_ground_truth.append(data[i][1])
        data[i] = ([word2idx[t] for t in data[i][0]], [tag2idx[t] for t in data[i][1]])

    dataset_word, dataset_tag = [], []
    for data_tuple in data:
        dataset_word.append(data_tuple[0])
        dataset_tag.append(data_tuple[1])

    with torch.no_grad():
        for data_tuple in data:
            mask = attention_mask(1).to(device)
            inputs = torch.tensor([data_tuple[0]], dtype=torch.long).to(device)
            y = torch.tensor([data_tuple[1]], dtype=torch.long).to(device)
            tag_scores = model(inputs, mask)
            loss_function = nn.CrossEntropyLoss()
            tag_scores_ = tag_scores.view(-1, tag_scores.shape[2])
            y_ = y.view(y.shape[0] * y.shape[1])
            loss = loss_function(tag_scores_, y_)
            total_loss += loss
            tag_scores = tag_scores.cpu().detach()
            pred_raw = torch.argmax(tag_scores, dim=2)
            pred = [idx2tag[str(int(t))] for t in pred_raw[0]]
            tag_prediction.append(pred)

    print("Epoch {} Validation loss: ".format(epoch+1), float(total_loss / num_of_batches))
    print(classification_report(tag_ground_truth, tag_prediction))

    return f1_score(tag_ground_truth, tag_prediction), float(total_loss / num_of_batches)


def train_transformer():
    word2idx = word_to_idx([training_data_path, validation_data_path, test_data_path])
    tag2idx, idx2tag = tag_to_idx(training_data_path)

    for i in range(len(training_data)):
        training_data[i] = ([word2idx[t] for t in training_data[i][0]], [tag2idx[t] for t in training_data[i][1]])

    dataset_word = []
    dataset_tag = []
    for data_tuple in training_data:
        dataset_word.append(data_tuple[0])
        dataset_tag.append(data_tuple[1])

    dataset_torch = Data.TensorDataset(torch.tensor(dataset_word, dtype=torch.long), torch.tensor(dataset_tag, dtype=torch.long))
    dataloader = Data.DataLoader(dataset=dataset_torch, batch_size=batch_size, shuffle=True, num_workers=2)
    model = TransformerTagger(len(word2idx), len(tag2idx), embedding_dim, nhead, hidden_dim, nlayers, dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    eval_loss, epoch_list, train_loss, f1s_list = [], [], [], []
    min_valid_loss = float('inf')

    for epoch in range(epochs):
        training_loss = 0.0
        with tqdm(dataloader, desc='Epoch {}'.format(epoch + 1)) as pbar:
            for i, (batch_x, batch_y) in enumerate(pbar):
                mask = attention_mask(len(batch_x)).to(device)
                model.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                tag_scores = model(batch_x, mask)
                tag_scores = tag_scores.view(-1, tag_scores.shape[2])
                batch_y = batch_y.view(batch_y.shape[0] * batch_y.shape[1])
                loss = loss_function(tag_scores, batch_y)
                training_loss += loss / len(dataloader)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                pbar.set_description("Epoch{}, Loss:{:.3}".format(epoch+1, loss))

        f1, valid_loss = transformer_eval(epoch, validation_data, model, word2idx, tag2idx, idx2tag, len(dataloader))
        epoch_list.append(epoch + 1)
        eval_loss.append(valid_loss)
        f1s_list.append(f1)
        training_loss = training_loss.cpu()
        train_loss.append(training_loss.detach())

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), "Transformer_{}_{}_{}_{}.pth".format(args.batch_size, args.dropout, args.embedding_dim, args.hidden_dim))


    plt.plot(epoch_list, eval_loss, label='Validation loss')
    plt.plot(epoch_list, train_loss, label='Training loss')
    plt.legend(loc="upper left")
    plt.grid(True, axis='y')
    plt.savefig("loss_transformer_{}_{}_{}_{}.png".format(args.batch_size, args.dropout, args.embedding_dim, args.hidden_dim))

    plt.figure()
    plt.plot(epoch_list, f1s_list, label='F1 score')
    plt.legend(loc="upper left")
    plt.grid(True, axis='y')
    plt.savefig("f1_transformer_{}_{}_{}_{}.png".format(args.batch_size, args.dropout, args.embedding_dim, args.hidden_dim))

if __name__ == "__main__":
    train_transformer()