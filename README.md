```python
# imports the various library for the lab
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import osmnx as ox # this line imports osmnx
import networkx as nx # this line imports networkx
import matplotlib.cm as cm
import matplotlib.colors as colors
#from IPython.display import IFrame
#ox.config(log_console=True, use_cache=True)

import sys
print (f'current environment: {sys.prefix}')

if ox.__version__=='2.0.1':
    #prints OSMNx version 
    print (f'current osmnx version: {ox.__version__}') 
else:
    #recommends student to upgrade to newer osmnx version.
    print (f'current osmnx version: {ox.__version__}. student might need to upgrade to osmnx=2.0.1 for the notebook to work')
```

    current environment: C:\Users\ASUS\anaconda3\envs\envGEOG0051
    current osmnx version: 2.0.1
    


```python
G2= ox.graph_from_address('Waterloo, London', dist=2000, network_type='drive')
ox.plot_graph(G2)
```


    
![png](Formative_files/Formative_1_0.png)
    





    (<Figure size 800x800 with 1 Axes>, <Axes: >)



## Example of Degree Centralilty analysis
(How many nodes connected to this point)



```python
G2= ox.graph_from_address('Waterloo, London', dist=2000, network_type='drive')
DG = ox.convert.to_digraph(G2) # Convert to diGraph

#Node Degree Centrality
node_dc = nx.degree_centrality(DG) # Calculate node degree Centrality
nx.set_node_attributes(DG, node_dc,'dc') # Set attributes bac to its node
G2 = nx.MultiGraph(DG) # Turn back to multiGraph for ploting
nc = ox.plot.get_node_colors_by_attr(G1, 'dc', cmap='plasma')
fig, ax = ox.plot_graph(G1, node_size=40, node_color=nc, 
                        edge_color='gray', edge_linewidth=0.5, edge_alpha=1)


#Edge Degree Centrality
edge_dc = nx.degree_centrality(nx.line_graph(DG))
nx.set_edge_attributes(DG, edge_dc,'dc') # Set attributes bac to its edge
G1 = nx.MultiGraph(DG)# and turn back to multiGraph for plotting
nc = ox.plot.get_edge_colors_by_attr(G1, 'dc', cmap='plasma')
fig, ax = ox.plot_graph(G1, node_size=0, node_color='w', node_edgecolor='gray', node_zorder=2,
                        edge_color=nc, edge_linewidth=1.5, edge_alpha=1) # Shows the edge centrality
```


```python

```


      Cell In[5], line 1
        jupyter nbconvert --to markdown <Formative.ipynb>
                ^
    SyntaxError: invalid syntax
    

