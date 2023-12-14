import argparse

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--path", type=str, default='./conll2003/')

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    return args