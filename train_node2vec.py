import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data


def load_adjacency_matrix(path):
    """
    Load adjacency matrix from specified path
    """
    print(f"Loading adjacency matrix from {path}")
    if path.endswith('.pkl'):
        return pd.read_pickle(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path, index_col=0)
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def adjacency_to_edge_index(adj_matrix):
    """
    Convert adjacency matrix to edge index format for PyTorch Geometric
    """
    if isinstance(adj_matrix, pd.DataFrame):
        node_names = adj_matrix.index
        node_index = {node: idx for idx, node in enumerate(node_names)}
        adj_np = adj_matrix.values
    else:
        adj_np = adj_matrix
        node_index = {i: i for i in range(adj_np.shape[0])}

    sources = []
    targets = []
    rows, cols = np.where(adj_np > 0)
    for i in range(len(rows)):
        sources.append(rows[i])
        targets.append(cols[i])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return edge_index, node_index


def train_node2vec(edge_index, num_nodes, embedding_dim=128, walk_length=30,
                   context_size=10, walks_per_node=10, p=1, q=1,
                   num_epochs=200, batch_size=128, learning_rate=0.001,
                   use_gpu=True):
    """
    Train a Node2Vec model using PyTorch Geometric
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Training on CPU")

    edge_index = edge_index.to(device)

    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        sparse=True
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=learning_rate)

    def train_epoch():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(num_epochs):
        loss = train_epoch()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    model.eval()
    with torch.no_grad():
        embeddings = model()

    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser(description='Train Node2Vec embeddings with different modalities')
    parser.add_argument('--modality', type=str, required=True, choices=['cond', 'icd', 'atc'],
                        help='Modality selector (cond or icd or med)')
    parser.add_argument('--output_dir', type=str, default='./data/',
                        help='Directory to save outputs')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of the embeddings')
    parser.add_argument('--walk_length', type=int, default=50,
                        help='Length of each random walk')
    parser.add_argument('--context_size', type=int, default=5,
                        help='Context window size')
    parser.add_argument('--walks_per_node', type=int, default=20,
                        help='Number of walks per node')
    parser.add_argument('--p', type=float, default=1.0,
                        help='Return parameter (1/p is probability of returning to source node)')
    parser.add_argument('--q', type=float, default=1.0,
                        help='In-out parameter (1/q is probability of moving outward)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimizer')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training even if GPU is available')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    matrix_path = f'./data/{args.modality}_adj_matrix.pkl'
    output_base = os.path.join(args.output_dir, f'{args.modality}_node_embeddings_{args.embedding_dim}')
    mapping_output = os.path.join(args.output_dir, f'{args.modality}_node_mapping_{args.embedding_dim}.csv')

    output_pt = f"{output_base}.pt"
    output_named = f"{output_base}_named.pkl"
    output_npy = f"{output_base}.npy"

    adj_matrix = load_adjacency_matrix(matrix_path)
    edge_index, node_index = adjacency_to_edge_index(adj_matrix)
    print(f"Graph has {len(node_index)} nodes and {edge_index.size(1)} edges")

    print(f"Training Node2Vec with embedding dimension {args.embedding_dim}")
    embeddings = train_node2vec(
        edge_index,
        num_nodes=len(node_index),
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        p=args.p,
        q=args.q,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_gpu=not args.cpu
    )

    print(f"Saving embeddings to {output_pt}")
    torch.save(embeddings, output_pt)
    np.save(embeddings, output_npy)

    if isinstance(adj_matrix, pd.DataFrame):
        reversed_mapping = {idx: node for node, idx in node_index.items()}
        mapping_df = pd.DataFrame(list(reversed_mapping.items()), columns=['Index', 'Node'])
        mapping_df.to_csv(mapping_output, index=False)
        print(mapping_df)
        print(f"Node mapping saved to {mapping_output}")

        named_embeddings = embeddings.detach().cpu().numpy()
        pd.to_pickle(named_embeddings, output_named)
        print(f"Named embeddings saved to {output_named}")


if __name__ == "__main__":
    main()