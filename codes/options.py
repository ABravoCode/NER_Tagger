import argparse

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--path", type=str, default='./conll2003/', help="path to data")

    parser.add_argument("--embedding_dim", type=int, default=128, help="embedding/padding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="number of heads for transformer")
    parser.add_argument("--nlayers", type=int, default=4, help="number of layers for transformer")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.05, help="momentum for SGD")
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    return args