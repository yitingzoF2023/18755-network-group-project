import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_induced_subgraph(file_path, seed_size=100, expansion_steps=2):
    DG = nx.DiGraph()
    
    print("Reading CSV file...")
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            DG.add_edge(row[0], row[1])
    
    print(f"Total nodes: {DG.number_of_nodes()}")
    print(f"Total edges: {DG.number_of_edges()}")
    
    # Select seed nodes
    seed_nodes = random.sample(list(DG.nodes()), seed_size)
    subgraph_nodes = set(seed_nodes)
    
    # Expand the subgraph
    for _ in range(expansion_steps):
        neighbors = set()
        for node in subgraph_nodes:
            neighbors.update(DG.predecessors(node))
            neighbors.update(DG.successors(node))
        subgraph_nodes.update(neighbors)
    
    # Create the induced subgraph
    subgraph = DG.subgraph(subgraph_nodes).copy()
    
    print(f"Induced subgraph nodes: {subgraph.number_of_nodes()}")
    print(f"Induced subgraph edges: {subgraph.number_of_edges()}")
    
    return subgraph

def analyze_network(DG):
    print(f"Number of nodes: {DG.number_of_nodes()}")
    print(f"Number of edges: {DG.number_of_edges()}")
    
    # Basic network metrics
    print(f"\nAverage clustering coefficient: {nx.average_clustering(DG):.4f}")
    print(f"Network density: {nx.density(DG):.4f}")
    
    # Degree distribution and CCDF
    in_degrees = [d for n, d in DG.in_degree()]
    out_degrees = [d for n, d in DG.out_degree()]
    total_degrees = [d for n, d in DG.degree()]
    
    plot_degree_distribution(in_degrees, "In-degree")
    plot_degree_distribution(out_degrees, "Out-degree")
    plot_degree_distribution(total_degrees, "Total degree")
    
    # Centrality measures
    print_top_nodes(nx.in_degree_centrality(DG), "in-degree centrality")
    print_top_nodes(nx.out_degree_centrality(DG), "out-degree centrality")
    print_top_nodes(nx.betweenness_centrality(DG), "betweenness centrality")
    print_top_nodes(nx.pagerank(DG), "PageRank")
    
    # Component analysis
    component_sizes = [len(c) for c in nx.weakly_connected_components(DG)]
    print(f"\nNumber of weakly connected components: {len(component_sizes)}")
    print(f"Size of the largest component: {max(component_sizes)}")
    print(f"Average component size: {np.mean(component_sizes):.2f}")

def plot_degree_distribution(degrees, degree_type):
    degree_count = Counter(degrees)
    x = sorted(degree_count.keys())
    y = [degree_count[d] for d in x]
    ccdf = [sum(y[i:]) / sum(y) for i in range(len(y))]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(x, ccdf, 'bo-', alpha=0.7)
    plt.xlabel(degree_type)
    plt.ylabel('CCDF')
    plt.title(f'CCDF of {degree_type} Distribution')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f'{degree_type.lower().replace(" ", "_")}_ccdf.png')
    plt.close()

def print_top_nodes(centrality_dict, centrality_type, n=5):
    print(f"\nTop {n} authors by {centrality_type}:")
    for author, centrality in sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]:
        print(f"{author}: {centrality:.4f}")

def visualize_graph(DG):
    pos = nx.spring_layout(DG, k=0.5, iterations=50)
    plt.figure(figsize=(20, 20))
    nx.draw(DG, pos, with_labels=False, node_size=30, node_color='lightblue', 
            edge_color='gray', alpha=0.6, linewidths=0.5, arrows=True, arrowsize=10)
    plt.title("Directed Coauthorship Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("directed_coauthorship_network.png")

if __name__ == "__main__":
    file_path = "dataset/coauthorship.csv"
    DG = load_induced_subgraph(file_path)
    analyze_network(DG)
    visualize_graph(DG)