import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
import math
import os
import matplotlib.pyplot as plt

# -------------------------
# Argument Parsing
# -------------------------
def ParseArgs():
    parser = argparse.ArgumentParser(description="DiffKG Hyperparameter Analysis")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--batch", default=64, type=int, help="Batch size")
    parser.add_argument("--epoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--tstEpoch", default=10, type=int, help="Test every n epochs")
    parser.add_argument("--latdim", default=64, type=int, help="Latent dimension size")
    parser.add_argument("--gnn_layers", default=2, type=int, help="Number of GNN layers")
    parser.add_argument("--topk", default=20, type=int, help="Top K items for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gnn_type", default="GCN", type=str, choices=["GCN", "GAT", "GIN"], help="Type of GNN")
    parser.add_argument("--variant", default="standard", type=str, choices=["standard", "w/o DM", "w/o CKGC"], help="Ablation Variant")
    parser.add_argument("--output_dir", default="results", type=str, help="Directory to save results")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for contrastive loss")
    parser.add_argument("--inf_steps", type=int, default=50, help="Number of diffusion inference steps")
    return parser.parse_args()

args = ParseArgs()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

class KGDataset(Dataset):
    def __init__(self, kg_tensor):
        self.kg_tensor = kg_tensor

    def __len__(self):
        return self.kg_tensor.size(0)

    def __getitem__(self, idx):
        return self.kg_tensor[idx]


# -------------------------
# Data Handling
# -------------------------
class DataHandler:
    def __init__(self):
        self.user_item_file = 'drug_phenotype_relationships.csv'
        self.kg_file = 'cancer_5hop_subgraph.csv'

    def LoadData(self):
        user_item_df = pd.read_csv(self.user_item_file)
        print("Columns in CSV:", user_item_df.columns)

        users = user_item_df['Phenotype'].unique()
        items = user_item_df['Drug'].unique()
        user2id = {user: idx for idx, user in enumerate(users)}
        item2id = {item: idx for idx, item in enumerate(items)}
        self.num_users = len(users)
        self.num_items = len(items)
        user_ids = user_item_df['Phenotype'].map(user2id).values
        item_ids = user_item_df['Drug'].map(item2id).values
        user_item_df['label'] = user_item_df['drug_relation'].apply(
            lambda rel: 1 if rel in ["Indication", "Drug Effect", "Off-label Use"] else 0
        )

        interactions = user_item_df['label'].values
        interaction_matrix = sp.coo_matrix((interactions, (user_ids, item_ids)), shape=(self.num_users, self.num_items))

        trnMat, tstMat = self.train_test_split(interaction_matrix)
        self.trnMat = trnMat
        self.tstMat = tstMat
        self.ui_matrix = self.buildSparseTensor(trnMat)
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.trnLoader = DataLoader(TrnData(trnMat), batch_size=args.batch, shuffle=True, num_workers=0)
        self.tstLoader = DataLoader(TstData(tstMat, trnMat), batch_size=args.batch, shuffle=False, num_workers=0)

        kg_df = pd.read_csv(self.kg_file)
        entities = pd.concat([kg_df['entity1'], kg_df['entity2']]).unique()
        self.num_entities = len(entities)
        self.entity2id = {entity: idx for idx, entity in enumerate(entities)}
        kg_df['head'] = kg_df['entity1'].map(self.entity2id).astype(int)
        kg_df['tail'] = kg_df['entity2'].map(self.entity2id).astype(int)
        relation_set = kg_df['relation'].unique()
        relation_mapping = {rel: idx for idx, rel in enumerate(relation_set)}
        kg_df['relation'] = kg_df['relation'].map(relation_mapping).astype(int)
        self.kg_triplets = kg_df[['head', 'relation', 'tail']].values


    def train_test_split(self, interaction_matrix, test_ratio=0.2, seed=42):
        np.random.seed(seed)
        interaction_data = np.array(interaction_matrix.data)
        row_indices = np.array(interaction_matrix.row)
        col_indices = np.array(interaction_matrix.col)
        indices = np.arange(len(interaction_data))
        np.random.shuffle(indices)
        split_point = int(len(interaction_data) * (1 - test_ratio))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        trnMat = sp.coo_matrix((interaction_data[train_indices], (row_indices[train_indices], col_indices[train_indices])), shape=interaction_matrix.shape)
        tstMat = sp.coo_matrix((interaction_data[test_indices], (row_indices[test_indices], col_indices[test_indices])), shape=interaction_matrix.shape)
        return trnMat, tstMat

    def buildSparseTensor(self, interaction_matrix):
        coo = interaction_matrix.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(args.device)
    def makeTorchAdj(self, interaction_matrix):
        num_users, num_items = interaction_matrix.shape
        zero_users = sp.coo_matrix((num_users, num_users))
        zero_items = sp.coo_matrix((num_items, num_items))
        adj_matrix = sp.bmat([[zero_users, interaction_matrix], [interaction_matrix.T, zero_items]])
        adj_matrix = adj_matrix.tocoo()
        identity = sp.identity(adj_matrix.shape[0], format='coo')
        adj_matrix = adj_matrix + identity
        adj_matrix = adj_matrix.tocoo()
        indices = np.vstack((adj_matrix.row, adj_matrix.col))
        indices = torch.tensor(indices, dtype=torch.long)
        values = torch.tensor(adj_matrix.data, dtype=torch.float32)
        shape = (num_users + num_items, num_users + num_items)

        return torch.sparse_coo_tensor(indices, values, shape).to(args.device)




# -------------------------
# Generative Model
# -------------------------

class GenerativeModel(nn.Module):
    def __init__(self, handler):
        super(GenerativeModel, self).__init__()
        self.handler = handler
        if args.variant == "w/o DM":
            self.model = VariationalGraphAutoencoder(handler.num_entities, args.latdim)
        else:
            self.model = GaussianDiffusionModel(handler.num_entities, args.latdim, args.inf_steps)
        
        self.entity_embedding = nn.Embedding(handler.num_entities, args.latdim)
        self.relation_embedding = nn.Embedding(len(handler.kg_triplets), args.latdim)

    def forward(self, x):
        head_embeds = self.entity_embedding(x[:, 0].long())
        rel_embeds = self.relation_embedding(x[:, 1].long())
        tail_embeds = self.entity_embedding(x[:, 2].long())
        triplet_embeds = torch.cat([head_embeds, rel_embeds, tail_embeds], dim=1)
        return self.model(triplet_embeds)


class VariationalGraphAutoencoder(nn.Module):
    def __init__(self, num_entities, latdim):
        super(VariationalGraphAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latdim * 3, latdim),  
            nn.ReLU(),
            nn.Linear(latdim, latdim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latdim, latdim * 3), 
            nn.ReLU(),
            nn.Linear(latdim * 3, latdim * 3), 
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)  
        x_recon = self.decoder(z)  
        return x_recon






# -------------------------
# Dataset Classes
# -------------------------
class TrnData(Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.negSampling()

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.dokmat.shape[1])
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class DiffKG(nn.Module):
    def __init__(self, handler):
        super(DiffKG, self).__init__()
        self.num_users = handler.num_users
        self.num_items = handler.num_items
        self.num_entities = handler.num_entities
        self.latdim = args.latdim
        self.uEmbeds = nn.Parameter(torch.randn(self.num_users, self.latdim).to(args.device))
        self.iEmbeds = nn.Parameter(torch.randn(self.num_items, self.latdim).to(args.device))
        self.eEmbeds = nn.Parameter(torch.randn(self.num_entities, self.latdim).to(args.device))
        self.gnnLayers = nn.ModuleList(
            [get_gnn_layer(args.gnn_type, self.latdim, self.latdim) for _ in range(args.gnn_layers)]
        )

        if args.variant != "w/o CKGC":
            self.kg_loss_weight = 1.0
            self.rel_embeds = nn.Embedding(len(handler.kg_triplets), self.latdim).to(args.device)
        else:
            self.kg_loss_weight = 0.0

    def forward(self, adj):
        embeds = torch.cat([self.uEmbeds, self.iEmbeds], dim=0)
        
        for layer in self.gnnLayers:
            embeds = layer(adj, embeds)
        
        userEmbeds = embeds[:self.num_users]
        itemEmbeds = embeds[self.num_users:]
        return userEmbeds, itemEmbeds

    def compute_kg_loss(self, kg_triplets):
        if args.variant == "w/o CKGC":
            return 0.0
        
        heads = torch.tensor(kg_triplets[:, 0], dtype=torch.long).to(args.device)
        relations = torch.tensor(kg_triplets[:, 1], dtype=torch.long).to(args.device)
        tails = torch.tensor(kg_triplets[:, 2], dtype=torch.long).to(args.device)
        
        head_embeds = self.eEmbeds[heads]
        tail_embeds = self.eEmbeds[tails]
        rel_embeds = self.rel_embeds(relations)
        scores = torch.norm(head_embeds + rel_embeds - tail_embeds, p=2, dim=1)
        kg_loss = torch.mean(scores)
        return kg_loss




class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.model = DiffKG(handler).to(args.device)  
        self.generative_model = GenerativeModel(handler).to(args.device) 
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.gen_opt = torch.optim.Adam(self.generative_model.parameters(), lr=args.lr)
        
        self.criterion = nn.BCELoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for user, pos_item, neg_item in self.handler.trnLoader:
            user = user.to(args.device).long()
            pos_item = pos_item.to(args.device).long()
            neg_item = neg_item.to(args.device).long()

            self.opt.zero_grad()
            userEmbeds, itemEmbeds = self.model(self.handler.torchBiAdj)
            pos_scores = (userEmbeds[user] * itemEmbeds[pos_item]).sum(dim=1)
            neg_scores = (userEmbeds[user] * itemEmbeds[neg_item]).sum(dim=1)

            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            kg_loss = self.model.compute_kg_loss(self.handler.kg_triplets)
            loss += kg_loss * self.model.kg_loss_weight
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.handler.trnLoader)
        return avg_loss
    

    def train_generative_model(self):
        self.generative_model.train()
        total_loss = 0

        kg_data = self.handler.kg_triplets
        kg_tensor = torch.tensor(kg_data, dtype=torch.long).to(args.device)

        kg_dataset = KGDataset(kg_tensor)
        kg_loader = DataLoader(kg_dataset, batch_size=args.batch, shuffle=True, num_workers=0)

        for data in kg_loader:
            data = data.to(args.device)
            self.gen_opt.zero_grad()

            recon = self.generative_model(data)

            head_embeds = self.generative_model.entity_embedding(data[:, 0].long())
            rel_embeds = self.generative_model.relation_embedding(data[:, 1].long())
            tail_embeds = self.generative_model.entity_embedding(data[:, 2].long())

            target = torch.cat([head_embeds, rel_embeds, tail_embeds], dim=1)
            loss = nn.MSELoss()(recon, target)

            loss.backward()
            self.gen_opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(kg_loader)
        return avg_loss




    def test_epoch(self):
        self.model.eval()
        total_recall, total_ndcg = 0, 0
        num_users = len(self.handler.tstLoader.dataset)

        with torch.no_grad():

            userEmbeds, itemEmbeds = self.model(self.handler.torchBiAdj)

            for user, trnMask in self.handler.tstLoader:
                user = user.long().to(args.device)
                trnMask = trnMask.to(args.device)
                scores = torch.matmul(userEmbeds[user], itemEmbeds.t())
                scores = scores * (1 - trnMask) - trnMask * 1e8
                _, topk_indices = torch.topk(scores, args.topk)

                recall, ndcg = self.calculate_metrics(
                    topk_indices.cpu().numpy(), 
                    self.handler.tstLoader.dataset.tstLocs, 
                    user.cpu().numpy()
                )
                total_recall += recall
                total_ndcg += ndcg

        avg_recall = total_recall / num_users
        avg_ndcg = total_ndcg / num_users
        return avg_recall, avg_ndcg

    def calculate_metrics(self, topk_indices, tstLocs, users):
        total_recall, total_ndcg = 0, 0
        for idx, user in enumerate(users):
            pred_items = list(topk_indices[idx])
            true_items = tstLocs[user]

            if len(true_items) == 0:
                continue

            hit_items = set(pred_items) & set(true_items)
            recall = len(hit_items) / len(true_items)
            dcg = sum([1.0 / math.log2(i + 2) for i, item in enumerate(pred_items) if item in true_items])
            idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(true_items), args.topk))])
            ndcg = dcg / idcg if idcg > 0 else 0.0

            total_recall += recall
            total_ndcg += ndcg

        return total_recall, total_ndcg

    def run(self):
        recalls = []
        ndcgs = []
        epochs = []

        with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
            for epoch in range(args.epoch):
                train_loss = self.train_epoch()
                gen_loss = self.train_generative_model()
                if epoch % args.tstEpoch == 0:
                    recall, ndcg = self.test_epoch()
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Gen Loss = {gen_loss:.4f}, Recall = {recall:.4f}, NDCG = {ndcg:.4f}")
                    f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Gen Loss = {gen_loss:.4f}, Recall = {recall:.4f}, NDCG = {ndcg:.4f}\n")
                    recalls.append(recall)
                    ndcgs.append(ndcg)
                    epochs.append(epoch)
                else:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Gen Loss = {gen_loss:.4f}")
                    f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Gen Loss = {gen_loss:.4f}\n")

            plt.figure()
            plt.plot(epochs, recalls, label='Recall@20')
            plt.plot(epochs, ndcgs, label='NDCG@20')
            plt.xlabel('Epochs')
            plt.ylabel('Performance')
            plt.legend()
            plt.title('Model Performance over Epochs')
            plt.savefig(os.path.join(args.output_dir, 'performance.png'))
            plt.close()


class TstData(Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
        tstLocs = [list() for _ in range(coomat.shape[0])]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            tstLocs[row].append(col)
            tstUsrs.add(row)
        self.tstUsrs = np.array(list(tstUsrs)) 
        self.tstLocs = tstLocs
    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        user = self.tstUsrs[idx]
        trnMask = np.reshape(self.csrmat[user].toarray(), [-1]) 
        return user, torch.FloatTensor(trnMask)






class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, adj, features):
        identity = torch.sparse_coo_tensor(
            indices=torch.arange(adj.shape[0]).repeat(2, 1),
            values=torch.ones(adj.shape[0], device=adj.device),
            size=(adj.shape[0], adj.shape[0]),
            device=adj.device
        )

        adj = adj + identity
        adj = adj.to_dense()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        output = torch.mm(norm_adj, features)
        output = self.linear(output)
        return self.activation(output)
    
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, epsilon=0.1):
        super(GINLayer, self).__init__()
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, adj, features):
        identity = torch.sparse_coo_tensor(
            indices=torch.arange(adj.shape[0]).repeat(2, 1),
            values=torch.ones(adj.shape[0], device=adj.device),
            size=(adj.shape[0], adj.shape[0]),
            device=adj.device
        )

        adj = adj + identity
        adj = adj.to_dense()

        row_sum = torch.sum(adj, dim=1)
        d_inv = torch.pow(row_sum, -1)
        d_inv[torch.isinf(d_inv)] = 0.0  
        d_mat_inv = torch.diag(d_inv)
        agg_neighbors = torch.mm(d_mat_inv, torch.mm(adj, features))

        out = (1 + self.epsilon) * features + agg_neighbors
        return self.mlp(out)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.1):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.attention_heads = nn.ModuleList([
            nn.Linear(2 * in_dim, 1) for _ in range(heads)
        ])
        self.linear = nn.Linear(in_dim, out_dim * heads)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, adj, features):
        identity = torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(adj.shape[0], device=adj.device),
                                 torch.arange(adj.shape[0], device=adj.device)]),
            values=torch.ones(adj.shape[0], device=adj.device),
            size=(adj.shape[0], adj.shape[0]),
            device=adj.device
        )
        adj = adj + identity

        adj_dense = adj.to_dense()

        N = features.shape[0]
        attention_scores = []
        for head in self.attention_heads:
            expanded_features = features.unsqueeze(1).repeat(1, N, 1)
            concatenated_features = torch.cat(
                [expanded_features, features.unsqueeze(0).repeat(N, 1, 1)],
                dim=-1
            )
            attention_score = torch.exp(head(concatenated_features).squeeze(-1))
            attention_scores.append(attention_score)
        attention_scores = torch.stack(attention_scores, dim=1)  
        attention_scores = attention_scores * adj_dense.unsqueeze(1)  
        attention_scores = attention_scores / (
            attention_scores.sum(dim=-1, keepdim=True) + 1e-10
        )
        attention_scores = self.dropout(attention_scores)

        feature_transformed = self.linear(features)
        feature_transformed = feature_transformed.view(N, self.heads, self.out_dim) 
        attention_output = torch.einsum("nhi,nij->nhj", attention_scores, feature_transformed)

        return self.activation(attention_output.view(N, -1)) 


class GaussianDiffusionModel(nn.Module):
    def __init__(self, num_entities, latdim, num_steps=50):
        super(GaussianDiffusionModel, self).__init__()
        self.num_steps = num_steps

        self.encoder = nn.Linear(latdim * 3, latdim)  
        self.decoder = nn.Linear(latdim, latdim * 3)  

        self.beta_schedule = torch.linspace(0.0001, 0.02, steps=num_steps).to(args.device)

    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = torch.prod(1 - self.beta_schedule[:t + 1]) 
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return noisy_x

    def reverse_diffusion(self, x_t, t):
        latent = F.relu(self.encoder(x_t)) 
        denoised = self.decoder(latent) 
        beta_t = self.beta_schedule[t]
        mean = (1 / torch.sqrt(1 - beta_t)) * (x_t - beta_t / torch.sqrt(1 - beta_t) * denoised)
        return mean

    def forward(self, x):
        for t in range(self.num_steps):
            x = self.forward_diffusion(x, t)
        for t in reversed(range(self.num_steps)):
            x = self.reverse_diffusion(x, t)
        return torch.sigmoid(x) 



# -------------------------
# Layer Getter
# -------------------------

def get_gnn_layer(gnn_type, in_dim, out_dim, heads=1):
    if gnn_type == "GCN":
        return GCNLayer(in_dim, out_dim)
    elif gnn_type == "GAT":
        return GATLayer(in_dim, out_dim, heads=heads)
    elif gnn_type == "GIN":
        return GINLayer(in_dim, out_dim)
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
# -------------------------
# Main Function
# -------------------------
def main():
    handler = DataHandler()
    handler.LoadData()
    print("Data loaded successfully.")
    coach = Coach(handler)
    print("Model initialized.")
    coach.run()
    print("Training and evaluation completed.")

if __name__ == "__main__":
    main()