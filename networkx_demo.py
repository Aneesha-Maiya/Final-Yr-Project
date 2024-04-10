# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# # Your 2D matrix
# matrix = [[0.         , 0.22995974, 0.21426465, 0.21881223, 0.24863189, 0.30203964],
#           [0.2158607 , 0.         , 0.24865982, 0.23579902, 0.22327803, 0.20712842],
#           [0.16247026, 0.20086635, 0.         , 0.20887688, 0.15383313, 0.14561883],
#           [0.17274207, 0.19831097, 0.21746708, 0.         , 0.16706371, 0.16101825],
#           [0.18970194, 0.18148433, 0.15478949, 0.16146208, 0.         , 0.18419486],
#           [0.25922504, 0.18937861, 0.16481896, 0.17504979, 0.20719324, 0.        ]]

# # Convert the matrix into a NetworkX graph
# G = nx.Graph()

# # Add nodes to the graph
# num_nodes = len(matrix)
# G.add_nodes_from(range(num_nodes))

# # Add edges to the graph with their weights
# for i in range(num_nodes):
#     for j in range(i+1, num_nodes):
#         weight = matrix[i][j]
#         if weight != 0:
#             G.add_edge(i, j, weight=weight)

# # Draw the graph
# pos = nx.spring_layout(G)  # positions for all nodes
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)
# edge_labels = {(i, j): round(d['weight'], 2) for i, j, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# plt.show()
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Your 2D matrix
matrix = [[0.         , 0.22995974, 0.21426465, 0.21881223, 0.24863189, 0.30203964],
          [0.2158607 , 0.         , 0.24865982, 0.23579902, 0.22327803, 0.20712842],
          [0.16247026, 0.20086635, 0.         , 0.20887688, 0.15383313, 0.14561883],
          [0.17274207, 0.19831097, 0.21746708, 0.         , 0.16706371, 0.16101825],
          [0.18970194, 0.18148433, 0.15478949, 0.16146208, 0.         , 0.18419486],
          [0.25922504, 0.18937861, 0.16481896, 0.17504979, 0.20719324, 0.        ]]

# Convert the matrix into a NetworkX graph
G = nx.Graph()

# Add nodes to the graph
num_nodes = len(matrix)
G.add_nodes_from(range(num_nodes))

# Add edges to the graph with their weights
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        weight = matrix[i][j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)

# Define edge thickness levels based on weight ranges
thickness_levels = {
    (0.0, 0.1): 1,  # thickness for weights in the range (0.0, 0.1)
    (0.1, 0.2): 2,  # thickness for weights in the range (0.1, 0.2)
    (0.2, 0.3): 4,  # thickness for weights in the range (0.2, 0.3)
    (0.3, 0.4): 6   # thickness for weights in the range (0.3, 0.4)
}

# Draw the graph with different edge thickness
pos = nx.spring_layout(G)  # positions for all nodes
for (start, end, weight) in G.edges(data='weight'):
    thickness = 1  # default thickness
    for (low, high), level in thickness_levels.items():
        if low < weight <= high:
            thickness = level
            break
    nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], width=thickness)

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)
edge_labels = {(i, j): round(d['weight'], 2) for i, j, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

plt.show()
