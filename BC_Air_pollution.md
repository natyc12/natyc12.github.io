# Network Centrality Serving as an indicator for Air Pollution in London
Select a neighbourhood of London you know well and get the graph in osmnx with a radius of 2000m. 1 You should use
no more than 2-3 analysis of the following measures calculated with osmnx and evaluate their effectiveness in representing
real-world aspects and identifying urban design issues within the selected area. Visualise and read the plots. The goal is to
create a gitpage to run a street network analysis for a neighbourhood in London.

What relationship can you find between retail locations with street network analysis? or between retail locations with tube
station or bus stations? If you were ask to place a shop to maximise footfall, where would you most likely want to place it? You
are also welcome to suggest your own question. Please include the following sections in your Gitpage:
• Introduction to describe your research question, for example, does closeness centrality or tube location affect location of
coffee shops in my area?
• Describe the dataset and neighbourhood succinctly
• Analysis and visualisation (creating the plots)
• Discussion (explaining the plots)


## 1.0 Introduction:

### 1.1 Research question:
To what extent does edge betweeness centrality and bus stop locations serve as an indicator for air pollution in Waterloo, London 

This study explores the edge betweeness centrality of the driving network within a 2000m bounding box of Waterloo in London. The primary source of air pollutants such as NO$_2$, NO$_{x}$, PM$_{2.5}$ and PM$_{10}$ in urban areas originate from road transport, thus will be explored in this study. Using data from the London Atmospheric Emissions Inventory (LAEI), the edge betweeness centrality of the network as an indicator for air pollution will be explored. Roads (edge) with high betweeness centrality indicate that it often lies on the shortest path between other nodes, acting as a connector in the overall network. This results in higher number of vehicles, thus should observe higher levels of pollution along these roads.




```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import osmnx as ox # this line imports osmnx
import networkx as nx # this line imports networkx
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import sys
import xyzservices.providers as xyz
from shapely.geometry import Point, box
G= ox.graph_from_address('Waterloo, London', dist=2000, network_type='drive')
```

## Describe the dataset and neighbourhood succinctly
Waterloo is located on the south bank of the River Thames in Central London. It is part of the London borough of Lambeth. Waterloo is connected to Strand area via the Waterloo Bridge. Within a 2000m area around Waterloo, it includes many notable places including Westminsiter, St James Park, Tottenham Court Road and St Paul's. There are multiple bridges that connect the North and South bank of the river within the study area, including the Vauxhall bridge, Lambeth Bridge, Westminster Bridge, Blackfriars Bridge, Southwark Bridge, and London Bridge.

The network data used is the driving network within the study area, with the extent of about (528950.7547,177899.0779) to (533046.34079,181955.8694). The air pollution data is retrieved from the 2019 London atmospheric Emissions inventory, with a data resolution of 20m. All values are an annual measure of pollutant concentration, in µg/m³.



```python
nodes, edges = ox.graph_to_gdfs(G)

# Reproject both GeoDataFrames to EPSG:3857 for compatibility with basemaps
nodes_proj = nodes.to_crs(epsg=3857)
edges_proj = edges.to_crs(epsg=3857)

# Create the plot
fig, ax = plt.subplots(figsize=(7, 7))

# Plot edges and nodes
edges_proj.plot(ax=ax, linewidth=1, edgecolor='black', zorder=2)
nodes_proj.plot(ax=ax, color='red', markersize=10, zorder=3)

# Set the limits to the extent of the nodes
minx, miny, maxx, maxy = nodes_proj.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Add the basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Remove axes
ax.set_axis_off()
plt.tight_layout()
plt.show()
```


    
![png](output_5_0.png)
    


## Analysis and Visualisation (creating the plots)

### Compute  betweeness centrality


```python
G= ox.graph_from_address('Waterloo, London', dist=2000, network_type='drive')
DG = ox.convert.to_digraph(G)
edge_bc = nx.betweenness_centrality(nx.line_graph(DG))
nx.set_edge_attributes(DG,edge_bc,'bc')
G1 = nx.MultiGraph(DG)
gdf_edges = ox.graph_to_gdfs(G1,nodes=False,fill_edge_geometry=True)
nodes, _ = ox.graph_to_gdfs(G)
minx, miny, maxx, maxy = nodes.total_bounds 
```

### Visualise Betweeness Centrality


```python
gdf_edges = gdf_edges.to_crs(epsg=27700)
nodes, _ = ox.graph_to_gdfs(G)
minx, miny, maxx, maxy = gdf_edges.total_bounds 


# Plot edges according to betweenness centrality, with a legend
ax = gdf_edges.plot(column='bc', cmap='plasma', figsize=(10,10), legend=True,
                      legend_kwds={'label': "Edge Betweenness Centrality",
                                   'orientation': "vertical"})
ax.set_xlim(minx-12, maxx)
ax.set_ylim(miny-70, maxy)
ctx.add_basemap(ax, crs="EPSG:27700", source=ctx.providers.CartoDB.Positron)
plt.axis('on')
plt.show()
```

    x-extent: 528950.7547472976 to 533046.3407566659
    y-extent: 177899.07793177164 to 181955.86948397692
    


    
![png](output_10_1.png)
    



```python
ax = gdf_edges.plot(column='bc', cmap='Oranges', figsize=(10,10), legend=True,scheme='equalinterval',k=5)
ax.set_xlim(minx-12, maxx)
ax.set_ylim(miny-70, maxy)

plt.axis('on')
plt.show()
```


    
![png](output_11_0.png)
    


The bridges connecting the north and south banks observed a high betweeness centrality (BC). This is excpected as they essentially act as a connector between nodes from the north to the south bank. High BC can also be observed along the north bank of the river, facilitaing the movement from east to west and vice versa. South of Waterloo, high BC roundabout could be seen. That is the roundabout of Elephant and Castle, providing connections to South of London

### Load CSV of air pollution
- Clean dataset: Remove points outside of the bounding box of 2000m radius from Waterloo
- Remove outliers
- Reproject


```python
# Read in the CSV files
df_no2 = pd.read_csv('data/NO2.csv')
df_nox = pd.read_csv('data/NOx.csv')
df_pm10 = pd.read_csv('data/PM10.csv')
df_pm25 = pd.read_csv('data/PM25.csv')

# Rename the 'conc' column to the pollutant name in each DataFrame
df_no2 = df_no2.rename(columns={'conc': 'NO2'})
df_nox = df_nox.rename(columns={'conc': 'NOx'})
df_pm10 = df_pm10.rename(columns={'conc': 'PM10'})
df_pm25 = df_pm25.rename(columns={'conc': 'PM25'})

# Merge the DataFrames on 'x' and 'y' (using inner join to retain only common points)
df = pd.merge(df_no2, df_nox, on=['x', 'y'], how='inner')
df = pd.merge(df, df_pm10, on=['x', 'y'], how='inner')
df = pd.merge(df, df_pm25, on=['x', 'y'], how='inner')

# Convert the DataFrame to a GeoDataFrame using the x and y columns, setting the CRS to BNG (EPSG:27700)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:27700")

# Retrieve the spatial bounds of graph G 
nodes, _ = ox.graph_to_gdfs(G)
minx, miny, maxx, maxy = nodes.total_bounds


# Create a polygon from these bounds and reproject it to BNG (EPSG:27700)
bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs=nodes.crs)
bbox_bng = bbox.to_crs(epsg=27700)
bminx, bminy, bmaxx, bmaxy = bbox_bng.total_bounds


gdf = gdf.cx[bminx:bmaxx, bminy:bmaxy] #gdf = Only include points within the bounding box of 2000m radius

import matplotlib.pyplot as plt

# List of columns to process
columns = ['NO2', 'NOx', 'PM10', 'PM25']

def remove_outliers(df, column, multiplier=1.5):
    """
    Removes outliers from the DataFrame column using the IQR method.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        column (str): The column name.
        multiplier (float): The multiplier for the IQR (default 1.5).
    
    Returns:
        DataFrame: A DataFrame with outliers removed for the given column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

air_gdf = remove_outliers(gdf, columns)

```

### Overlay Betweeness Centrality with Pollutants


```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mapclassify
import pandas as pd

# List of pollutant columns
columns = ['NO2', 'NOx', 'PM10', 'PM25']

# First, create a classification for BC over the whole edges GeoDataFrame.
# We'll use an EqualInterval scheme with 3 classes.
bc_classifier = mapclassify.EqualInterval(gdf_edges['bc'], k=3)
bc_bins = bc_classifier.bins
# Create labels that show the range for each bin (rounded to 2 decimals)
bc_min = gdf_edges['bc'].min()
labels_bc = [
    f"{round(bc_min, 2)} - {round(bc_bins[0], 2)}",
    f"{round(bc_bins[0], 2)} - {round(bc_bins[1], 2)}",
    f"{round(bc_bins[1], 2)} - {round(bc_bins[2], 2)}"
]
# Use a discrete version of the 'Blues' colormap with 3 colors.
cmap_bc = plt.get_cmap('Blues', 3)
patches_bc = [mpatches.Patch(color=cmap_bc(i), label=labels_bc[i]) for i in range(3)]

# Set up the subplots.
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

for i, col in enumerate(columns):
    ax = axs[i // 2, i % 2]
    
    # --- Pollutant layer ---
    # Remove outliers using your custom function.
    filtered_gdf = remove_outliers(gdf, col).copy()
    
    # Create 5 quantile groups for the pollutant.
    # We'll use pd.qcut to assign a quantile category.
    quantile_col = col + '_quantile'
    filtered_gdf[quantile_col] = pd.qcut(filtered_gdf[col], q=5, duplicates='drop')
    
    # Use mapclassify to get the bins and round them.
    pollutant_classifier = mapclassify.Quantiles(filtered_gdf[col].dropna(), k=5)
    pollutant_bins = pollutant_classifier.bins
    lower_bound_poll = filtered_gdf[col].min()
    # Build labels for each bin rounded to whole numbers.
    labels_poll = []
    for j, b in enumerate(pollutant_bins):
        if j == 0:
            labels_poll.append(f"{int(round(lower_bound_poll, 0))} - {int(round(b, 0))}")
        else:
            labels_poll.append(f"{int(round(pollutant_bins[j-1], 0))} - {int(round(b, 0))}")
    
    # Create a discrete colormap for pollutant using 'Reds'.
    cmap_poll = plt.get_cmap('Reds', len(labels_poll))
    patches_poll = [mpatches.Patch(color=cmap_poll(j), label=labels_poll[j]) 
                    for j in range(len(labels_poll))]
    
    # Plot the pollutant layer using the discrete quantile categories.
    filtered_gdf.plot(
        ax=ax, 
        column=quantile_col, 
        categorical=True, 
        cmap='Reds', 
        legend=False
    )
    
    # --- BC overlay ---
    # Plot the network edges with BC using an EqualInterval scheme into 3 classes.
    gdf_edges.plot(
        ax=ax, 
        column='bc', 
        cmap='Blues', 
        scheme='EqualInterval', 
        k=5, 
        legend=False
    )
    
    # Add the pollutant legend (patches) to the right of each subplot.
    ax.legend(handles=patches_poll, title=f"{col} Quantiles", loc='upper left', 
              bbox_to_anchor=(1.05, 1), frameon=False)
    
    # Remove axes and set spatial limits and title.
    ax.set_axis_off()
    ax.set_title(f"Plot for {col} with BC Overlay", pad=20)
    ax.set_xlim(bminx, bmaxx)
    ax.set_ylim(bminy, bmaxy)

# Add one overall BC legend (patches) for the entire figure.
fig.legend(
    handles=patches_bc, 
    title="Edge Betweenness Centrality\n(EqualInterval)", 
    frameon=False
)

plt.tight_layout()
plt.show()
fig.savefig('multiplot_draft.pdf', dpi=150)
```


    
![png](output_16_0.png)
    


Clear pattern of high pollution can be seen along roads. The roads with high BC is observed with higher air pollution. This is quite evident in PM10 and PM2.5 pollution, where high concentrations are obsrbed along the bridges, along the north bank, as well as the roundabout at Elephant and Castle. The dataset of NO2 and NOx were missing some data along the north bank of the river, however a clear dinstinction of concentrations along high BC roads vs low BC can be observed. This is especially clear in the south side. 

However, edge effects definetely played a role in skewing betweeness centrality measures. Soho area (North East corner of the study area) and Westminister (South of Soho) can be observed with high air pollution. However they fall in the lowest quantile for BC. They reside on the edges of the study area, thus in this calculation, they are observed with lower BC. In reality (when considering the whole network of London) they will display a high between centrality as they are situated in Central London.

There is potential for BC to act as an indicator as air pollution. Where furhter testing can be done, whilst omitting areas on the edge, a regression can be done to test their relationship. This relationship may not hold true in other areas were Land Use may come into play. E.g. Airports may not have high BC as they are situated in the peripherals of the city, but high air pollution. But also tourist hotspots for example, located in more peripheral areas might have high traffic = high pollution, whilst not having a high BC.
