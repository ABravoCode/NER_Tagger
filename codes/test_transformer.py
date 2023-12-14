import warnings
warnings.filterwarnings("ignore")

import copy
import torch
import torch.nn as nn
from model import TransformerTagger, generate_square_subsequent_mask
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils import *
from options import get_options
args = get_options()

embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
nlayers = args.nlayers
nhead = args.nhead
dropout = args.dropout
batch_size = args.batch_size
path = args.path
device = args.device

training_data_path = path + "train.txt"
validation_data_path = path + "valid.txt"
test_data_path = path + "test.txt"
training_data = dataset_build_with_batch(training_data_path, batch_size)
validation_data = dataset_build(validation_data_path)
testing_data, testing_middle = fullset_build(test_data_path)

word2idx = word_to_idx([training_data_path, validation_data_path, test_data_path])
tag2idx, idx2tag = tag_to_idx(training_data_path)

model = TransformerTagger(len(word2idx), len(tag2idx), embedding_dim, nhead, hidden_dim, nlayers, dropout).to(device)
model.load_state_dict(torch.load("Transformer_{}_{}_{}_{}.pth".format(args.batch_size, args.dropout, args.embedding_dim, args.hidden_dim)))

word2idx = word_to_idx([training_data_path, validation_data_path, test_data_path])
tag2idx, idx2tag = tag_to_idx(training_data_path)

with torch.no_grad():
    data = copy.deepcopy(testing_data)
    pred_right = 0
    total_num = 0
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
        print("Test Transformer model.")
        for data_tuple in data:
            mask = generate_square_subsequent_mask(1).to(device)
            inputs = torch.tensor([data_tuple[0]], dtype=torch.long).to(device)
            y = torch.tensor([data_tuple[1]], dtype=torch.long).to(device)
            tag_scores = model(inputs, mask)
            loss_function = nn.CrossEntropyLoss()

            tag_scores_ = tag_scores.view(-1, tag_scores.shape[2])
            y_ = y.view(y.shape[0] * y.shape[1])

            loss = loss_function(tag_scores_, y_)
            total_loss += loss

            if torch.cuda.is_available():
                tag_scores = tag_scores.cpu().detach()
            pred_raw = torch.argmax(tag_scores, dim=2)
            pred = [idx2tag[str(int(t))] for t in pred_raw[0]]
            tag_prediction.append(pred)

    print(classification_report(tag_ground_truth, tag_prediction))
    print("Accuracy: ", accuracy_score(tag_ground_truth, tag_prediction))
    print("Precision: ", precision_score(tag_ground_truth, tag_prediction))
    print("Recall: ", recall_score(tag_ground_truth, tag_prediction))
    print("F1 score: ", f1_score(tag_ground_truth, tag_prediction))


f = open('output_transformer.txt', 'w+')
f1 = open('./conll2003/test.txt', 'r')
for i in range(len(testing_data)):
    for j in range(len(testing_data[i][0])):
        f.write('{} {} {} {}\n'.format(testing_data[i][0][j], testing_middle[i][j][0], testing_middle[i][j][1], tag_prediction[i][j]))

    f.write('\n')


f1.close()
f.close()