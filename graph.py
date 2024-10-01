import csv
import networkx as nx
import matplotlib.pyplot as plt

def create_coauthorship_graph(file_path):
    G = nx.DiGraph()
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            author1, author2 = row[0], row[1]
            G.add_edge(author1, author2)
    
    return G

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
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', 
            font_size=8, font_weight='bold', arrows=True)
    plt.title("Coauthorship Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "dataset/coauthorship.csv"
    G = create_coauthorship_graph(file_path)
    # analyze_network(G)
    visualize_graph(G)