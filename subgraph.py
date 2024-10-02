import csv
import random
import networkx as nx
import matplotlib.pyplot as plt

def load_subset_graph(file_path, sample_size=10000, edge_sample_size=100000):
    G = nx.Graph()
    authors = set()
    edges = []
    
    print("Reading CSV file...")
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            authors.add(row[0])
            authors.add(row[1])
            edges.append((row[0], row[1]))
    
    print(f"Total unique authors: {len(authors)}")
    print(f"Total edges: {len(edges)}")
    
    sampled_authors = set(random.sample(list(authors), min(sample_size, len(authors))))
    print(f"Sampled authors: {len(sampled_authors)}")
    
    sampled_edges = random.sample(edges, min(edge_sample_size, len(edges)))
    print(f"Sampled edges: {len(sampled_edges)}")
    
    for author1, author2 in sampled_edges:
        if author1 in sampled_authors or author2 in sampled_authors:
            G.add_edge(author1, author2)
    
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    
    # Keep the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc).copy()
    
    print(f"Largest component nodes: {subgraph.number_of_nodes()}")
    print(f"Largest component edges: {subgraph.number_of_edges()}")
    
    return subgraph

def analyze_network(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    print("\nTop 5 authors by degree centrality:")
    for author, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{author}: {centrality:.4f}")
    
    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    print("\nTop 5 authors by betweenness centrality:")
    for author, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{author}: {centrality:.4f}")
    
    # PageRank
    pagerank = nx.pagerank(G)
    print("\nTop 5 authors by PageRank:")
    for author, rank in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{author}: {rank:.4f}")

def visualize_graph(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(20, 20))
    nx.draw(G, pos, with_labels=False, node_size=30, node_color='lightblue', 
            edge_color='gray', alpha=0.6, linewidths=0.5)
    plt.title("Coauthorship Network (Largest Connected Component)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "dataset/coauthorship.csv"
    G = load_subset_graph(file_path)
    analyze_network(G)
    visualize_graph(G)