import pandas as pd
from collections import deque, defaultdict

pharmkg_train_path = 'data/train.tsv'
pharmkg_train = pd.read_csv(pharmkg_train_path, sep='\t', header=None, names=['entity1', 'relation', 'entity2'])
pharmkg_train = pharmkg_train.drop_duplicates().reset_index(drop=True)
cancer_keywords = ['cancer', 'breast', 'cervical', 'carcinoma', 'tumor']
def contains_cancer_keyword(entity):
    return any(keyword.lower() in str(entity).lower() for keyword in cancer_keywords)
all_entities = pd.concat([pharmkg_train['entity1'], pharmkg_train['entity2']]).unique()
cancer_entities = set(filter(contains_cancer_keyword, all_entities))
print(f"Number of initial cancer-related entities: {len(cancer_entities)}")
adjacency_list = defaultdict(list)
for idx, row in pharmkg_train.iterrows():
    adjacency_list[row['entity1']].append((row['entity2'], row['relation']))
    adjacency_list[row['entity2']].append((row['entity1'], row['relation']))  # Assuming undirected graph

def bfs(adjacency_list, start_nodes, max_depth):
    visited_nodes = set()
    visited_edges = set()
    queue = deque()
    for node in start_nodes:
        queue.append((node, 0))
        visited_nodes.add(node)
    while queue:
        current_node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        
        for neighbor, relation in adjacency_list[current_node]:
            edge = (current_node, relation, neighbor)
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, depth + 1))
            if edge not in visited_edges:
                visited_edges.add(edge)
    
    return visited_nodes, visited_edges

max_depth = 5
visited_nodes, visited_edges = bfs(adjacency_list, cancer_entities, max_depth)
print(f"Total nodes in 5-hop neighborhood: {len(visited_nodes)}")
print(f"Total edges in 5-hop neighborhood: {len(visited_edges)}")
subgraph_edges = pd.DataFrame(list(visited_edges), columns=['entity1', 'relation', 'entity2'])
print(subgraph_edges.head())
subgraph_edges.to_csv('cancer_5hop_subgraph.csv', index=False)