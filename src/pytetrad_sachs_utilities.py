"""
Description: Utility functions for working with the Sachs dataset, Tetrad, and causallearn tool kits.

Author: Chris Andersen
Date: 12/11/2024

"""
import re
import pandas as pd
import numpy as np

from pytetrad.tools import translate as tr
import edu.cmu.tetrad.graph as graph
import java.io as _io
import edu.cmu.tetrad.algcomparison.statistic as stat

# import causal learn functions
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

def baseline_models():
    """
    Defines the various baseline models used in our analysis.
    """
    # Sachs baseline edges
    baseline_edges = [
        "pkc --> raf",
        "pkc --> p38",
        "pkc --> jnk",
        "plc --> pkc",
        "pip2 --> pkc",
        "raf --> mek",
        "pka --> raf",
        "mek --> erk",
        "pka --> erk",
        "pka --> p38",
        "pka --> jnk",
        "pka --> akt",
        "pip3 --> akt",
        "pip3 --> pip2",
        "pip2 --> pip3",
        "plc --> pip2",
        "pip3 --> plc"
    ]

    # Sachs ground truth with <85% edges (ramsey et al 2018, Figure 3)
    ramsey_ground_truth = [
        "akt o-o raf",
        "akt o-o plc",
        "akt o-o pkc",
        "akt o-o mek",
        "erk --> akt",
        "jnk o-o mek",
        "jnk o-o p38",
        "mek --> erk",
        "mek o-o plc",
        "pip2 --> pkc",
        "pip3 --> pip2",
        "pip3 --> plc",
        "pip3 --> akt",
        "pka o-o pkc",
        "pka o-o plc",
        "pka --> raf",
        "pka --> akt",
        "pka --> erk",
        "pka --> jnk",
        "pka --> mek",
        "pka --> p38",
        "pkc --> raf",
        "pkc o-o akt",
        "pkc --> jnk",
        "pkc --> mek",
        "pkc --> p38",
        "pkc o-o pka",
        "plc --> pkc",
        "plc o-o akt",
        "plc o-o mek",
        "plc --> pip2",
        "plc o-o pka",
        "raf o-o akt",
        "raf --> mek"
    ]

    # Sachs model according to Ramsey et al. 2018
    ramsey_fig_5 = [
        "erk --> akt",
        "mek --> erk",
        "pip3 --> pip2",
        "pka o-o pkc",
        "pka --> raf",
        "pka --> akt",
        "pka --> erk",
        "pka --> jnk",
        "pka --> mek",
        "pka --> p38",
        "pkc --> raf",
        "pkc --> jnk",
        "pkc --> mek",
        "pkc --> p38",
        "plc --> pip2",
        "plc --> pip3",
        "raf --> mek"
    ]

    # FASK estimated graph from Ramsey et al. 2018
    ramsey_fask = [
        "akt --> pka",
        "raf --> mek",
        "plc --> pip2",
        "pkc --> raf",
        "pkc --> jnk",
        "pkc --> mek",
        "pkc --> p38",
        "pka --> pkc",
        "pka --> plc",
        "pka --> raf",
        "pka --> akt",
        "pka --> erk",
        "pka --> jnk",
        "pka --> mek",
        "pka --> p38",
        "pka --> pip2",
        "pip3 --> plc",
        "pip3 --> pip2",
        "mek --> raf",
        "jnk --> mek",
        "erk --> akt"
    ]

    # create starting edge dataframe for each baseline vector
    gt_df = pd.DataFrame({"baseline_edges": baseline_edges})
    gt_df.sort_values(by="baseline_edges", inplace=True)
    gt_df.reset_index(drop=True, inplace=True)

    r_gt_df = pd.DataFrame({"ramsey_ground_truth": ramsey_ground_truth})
    r_gt_df.sort_values(by="ramsey_ground_truth", inplace=True)
    r_gt_df.reset_index(drop=True, inplace=True)

    r_fig_5_df = pd.DataFrame({"ramsey_fig_5": ramsey_fig_5})
    r_fig_5_df.sort_values(by="ramsey_fig_5", inplace=True)
    r_fig_5_df.reset_index(drop=True, inplace=True)

    r_fask_df = pd.DataFrame({"ramsey_fask": ramsey_fask})
    r_fask_df.sort_values(by="ramsey_fask", inplace=True)
    r_fask_df.reset_index(drop=True, inplace=True)

    return gt_df, r_gt_df, r_fig_5_df, r_fask_df

def write_gdot(g, gdot, node_mapping=None):
    """
    Write a Tetrad graph to a GraphViz format with optional node name translation.
    
    Parameters:
    -----------
    g : TetradGraph
        The graph object from Tetrad
    gdot : GraphViz.Graph
        The GraphViz graph object to write to
    node_mapping : dict, optional
        Dictionary mapping original node names to desired display names
        e.g., {'X1': 'akt', 'X2': 'erk', ...}
    """
    endpoint_map = {"TAIL": "none",
                   "ARROW": "empty",
                   "CIRCLE": "odot"}

    # Add graph-level attributes for better layout
    gdot.attr('graph', rankdir='TB')
    
    # Helper function to translate node names
    def get_display_name(node_name):
        if node_mapping and str(node_name) in node_mapping:
            return node_mapping[str(node_name)]
        return str(node_name)
    
    # First pass: add all nodes with translated names
    for node in g.getNodes():
        original_name = str(node.getName())
        display_name = get_display_name(original_name)
        
        gdot.node(original_name,  # Use original name as node identifier
                 label=display_name,  # Use translated name as display label
                 shape='circle',
                 fixedsize='true',
                 width='0.6',
                 height='0.6',
                 style='filled',
                 fontsize='12',
                 color='lightgray')

    # Second pass: add all edges (using original names as identifiers)
    for edge in g.getEdges():
        node1 = str(edge.getNode1().getName())
        node2 = str(edge.getNode2().getName())
        endpoint1 = str(endpoint_map[edge.getEndpoint1().name()])
        endpoint2 = str(endpoint_map[edge.getEndpoint2().name()])
        
        color = "blue"
        if (endpoint1 == "empty") and (endpoint2 == "empty"): 
            color = "red"
            
        gdot.edge(node1, node2,
                 arrowtail=endpoint1,
                 arrowhead=endpoint2,
                 dir='both', 
                 color=color,
                 penwidth='1.0')

    return gdot

def replace_nodes(string, node_dict, reverse=False):
    if reverse:
        for key, value in node_dict.items():
            # Use word boundaries (\b) to ensure exact matches
            string = re.sub(r'\b{}\b'.format(re.escape(value)), key, string)
    else:
        for key, value in node_dict.items():
            # Use word boundaries (\b) to ensure exact matches
            string = re.sub(r'\b{}\b'.format(re.escape(key)), value, string)
    return string

def add_edges_column(df, col, pattern="", out_str="", out_col=[]):
    """
    Adjust the size of a dataframe and add a new column with a list of edges.

    Parameters:
    - df: A dataframe containing the Sachs dataset.
    - col: The name of the new column to add to the dataframe.
    - pattern: A regular expression pattern to extract edges from the output string.
    - out_str: A string containing the output of a causal discovery algorithm. default: "".
    - out_col: A list of edges to add to the dataframe. default: [].

    Returns:
    - The updated dataframe with the new column added.
    """
    if out_str == "":
        # out_col should be non-empty list of edges
        edges_list = out_col
    elif out_col == []:
        edges = re.findall(pattern, out_str)
        edges_list = [' '.join(edge) for edge in edges]
    max_length = max(len(df), len(edges_list))
    df = df.reindex(range(max_length))
    if len(edges_list) < max_length:
        edges_list.extend([pd.NA]*(max_length - len(edges_list)))
    df[col] = edges_list
    return df

def create_baseline_model(df, nodes):
    """
    Create a baseline model for each edge in the Sachs dataset.

    Parameters:
    - df: A dataframe containing the Sachs dataset.
    - measured_cols: The list of nodes in the dataset.
    - baseline_col: The name of the column containing the baseline edges. default: "baseline_edges".

    Returns:
    - A dictionary containing the baseline model for each edge in the dataset.
    """
    # Define a regular expression pattern to capture all the edge types
    colname = df.columns[0]
    pattern = r" --> | o-o | <-> | o-> "
    edges = [re.split(pattern, edge) for edge in df[colname]]
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    node_index = {node: i for i, node in enumerate(nodes)}
    for source, target in edges:
        adj_matrix[node_index[source]][node_index[target]] = 1
    baseline_graph = tr.adj_matrix_to_graph(adj_matrix)
    tetrad_data = tr.pandas_data_to_tetrad(df)
    return baseline_graph, tetrad_data

def create_filtered_graph(fask_str_fil, measured_cols):
    """
    Create a new graph from a filtered FASK string output.
    
    Parameters:
    - fask_str_fil: The filtered string output from FASK (with intervention variables removed)
    - measured_cols: List of measured variables to include
    
    Returns:
    - A new graph object containing only the measured variables and their edges
    """
    # Define pattern for edge splitting
    pattern = r" --> | o-o | <-> | o-> "
    
    # Extract edges from the string
    # Assuming the edges come after "Graph Edges:" in the string
    edges_section = fask_str_fil.split("Graph Edges:\n")[1]
    edge_lines = [line.split(". ")[1] for line in edges_section.strip().split("\n")]
    edges = [re.split(pattern, edge) for edge in edge_lines]
    
    # Create adjacency matrix
    num_nodes = len(measured_cols)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    node_index = {node: i for i, node in enumerate(measured_cols)}
    
    for source, target in edges:
        adj_matrix[node_index[source]][node_index[target]] = 1
        
    # Convert back to graph
    filtered_graph = tr.adj_matrix_to_graph(adj_matrix)
    
    return filtered_graph

def compute_statistics(baseline_graph, model_str, data, node_names):
    """
    Compute statistics for a model compared to a baseline model.

    Parameters:
    - baseline_graph: The baseline model to compare against.
    - model_str: The string output of the causal discovery algorithm.
    - data: The data being tested.
    - node_names: A dictionary mapping node names to their corresponding indices.

    Returns:
    - A dictionary containing the adjacency precision, adjacency recall, arrowhead precision, and arrowhead recall.
    """
    model_str = replace_nodes(model_str, node_names, reverse=True)
    model = graph.GraphSaveLoadUtils.readerToGraphTxt(_io.StringReader(model_str))
    model = graph.GraphUtils.replaceNodes(model, baseline_graph.getNodes())
    ap = stat.AdjacencyPrecision().getValue(baseline_graph, model, data)
    ar = stat.AdjacencyRecall().getValue(baseline_graph, model, data)
    ahp = stat.ArrowheadPrecision().getValue(baseline_graph, model, data)
    ahr = stat.ArrowheadRecall().getValue(baseline_graph, model, data)
    return {
        'adjacency_precision': ap,
        'adjacency_recall': ar,
        'arrowhead_precision': ahp,
        'arrowhead_recall': ahr
    }

def remove_edges(graph_output, exclude_cols):
    """
    Filters edges from the graph output where either node is an intervention variable
    and re-indexes the remaining edges sequentially.

    Parameters:
    - graph_output: A string containing the graph edges as output by FASK.
    - exclude_cols: A list of node names to exclude from the graph.

    Returns:
    - A string containing the header, re-indexed edges, and any trailing lines.
    """
    # Regular expression to capture the edges and their nodes
    pattern = r'\d+\.\s+([\w_]+)\s+(-->|<->|o->|o-o|---)\s+([\w_]+)'

    # Split the string into lines
    lines = graph_output.splitlines()
    
    # List to store the filtered output (header, re-indexed edges, and trailing lines)
    filtered_output = []
    reindexed_edges = []  # For storing and re-indexing edges
    filter_edges = False  # Start filtering after the 'Graph Edges:' line
    trailing_lines = []  # To capture lines after the edge section ends

    # Iterate through each line in the graph output
    for line in lines:
        if "Graph Edges:" in line:
            filter_edges = True  # Start filtering edges after this line
            filtered_output.append(line)  # Keep the "Graph Edges:" line itself
            continue

        if filter_edges:
            match = re.match(pattern, line)
            if match:
                node1, _, node2 = match.groups()
                # Check if either node is an intervention variable
                if node1 not in exclude_cols and node2 not in exclude_cols:
                    reindexed_edges.append(line)  # Store edge for re-indexing
            else:
                # No longer in edge section, capture trailing lines
                trailing_lines.append(line)

        else:
            # Add header lines before "Graph Edges:"
            filtered_output.append(line)

    # Re-index the remaining edges
    for i, edge in enumerate(reindexed_edges, start=1):
        # Replace the original index with the new index
        reindexed_edge = re.sub(r'^\d+', str(i), edge)
        filtered_output.append(reindexed_edge)
    
    # Add trailing lines back into the output
    filtered_output.extend(trailing_lines)

    # Return the filtered and re-indexed output as a string
    return "\n".join(filtered_output)

def adjacency_to_graph_string(adjacency_matrix, node_names=None, edge_weight=False):
    """
    Convert a LiNGAM adjacency matrix to a graph string representation in TETRAD format.
    
    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        The adjacency matrix from LiNGAM where entry [i,j] represents 
        the edge weight from node j to node i
    node_names : list, optional
        List of node names. If None, will use X1, X2, etc.
    edge_weight : bool, optional
        Whether to include edge weights in the output
        
    Returns:
    --------
    str
        A string representation of the graph in TETRAD format
    """
    # Create the standard TETRAD header
    header = "None\n\nNone\n\n/knowledge\naddtemporal\n\n\nforbiddirect\n\nrequiredirect\n\n"
    
    n_nodes = adjacency_matrix.shape[0]
    
    # Create node names if not provided
    if node_names is None:
        node_names = [f'X{i}' for i in range(n_nodes)]
    
    # Create the nodes section
    nodes_str = ';'.join(node_names)  # Note: TETRAD format uses semicolons
    
    # Create the edges section
    edges = []
    edge_count = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency_matrix[i,j] != 0:
                edge_count += 1
                if edge_weight:
                    edges.append(f"{edge_count}. {node_names[j]} --> {node_names[i]} ({adjacency_matrix[i,j]:.3f})")
                else:
                    edges.append(f"{edge_count}. {node_names[j]} --> {node_names[i]}")
    
    edges_str = '\n'.join(edges)
    
    return f"{header}Graph Nodes:\n{nodes_str}\n\nGraph Edges:\n{edges_str}\n\n"

def adjacency_to_graph(adjacency_matrix, node_names=None):
    """
    Convert a LiNGAM adjacency matrix to a causallearn causal graph object.
    
    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        The adjacency matrix from LiNGAM where entry [i,j] represents 
        the edge weight from node j to node i
    node_names : list, optional
        List of node names
        
    Returns:
    --------
    CausalGraph
        A causallearn CausalGraph object representing the LiNGAM results
    """
    n_nodes = adjacency_matrix.shape[0]
    if node_names is None:
        node_names = [f'X{i+1}' for i in range(n_nodes)]
    
    # Create a causal graph instance with node names
    cg = CausalGraph(n_nodes, node_names)
    
    # Helper function to safely remove edges if they exist
    def remove_if_exists(x: int, y: int) -> None:
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)
    
    # First, remove all undirected edges from the initial complete graph
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Note: only need to check upper triangle
            remove_if_exists(i, j)
            remove_if_exists(j, i)
    
    # Then add directed edges based on non-zero entries in adjacency matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency_matrix[i,j] != 0:
                # Add directed edge from j to i (following LiNGAM's convention)
                cg.G.add_edge(Edge(cg.G.nodes[j], cg.G.nodes[i], 
                                 Endpoint.TAIL, Endpoint.ARROW))
    
    return cg

def add_background_knowledge(int_cols, measured_cols, obj = None, library="pytetrad"):
    """
    Add background knowledge to a TetradSearch object.
    
    Parameters:
    -----------
    int_cols : list
        List of intervention variables
    measured_cols : list
        List of measured variables
    obj : TetradSearch, optional
        A TetradSearch object to add background knowledge to
    library : str, optional
        The library to use. Either 'pytetrad' or 'causallearn'
    """
    if library == "pytetrad":
        if obj is None:
            raise ValueError("Must provide a TetradSearch object to add background knowledge.")
        # Add intervention variables to tier 0
        for var in int_cols:
            obj.add_to_tier(0, var)

        # Add measured proteins to tier 1
        for var in measured_cols:
            obj.add_to_tier(1, var)

        # Forbid protein to intervention edges
        for protein in measured_cols:
            for intervention in int_cols:
                obj.set_forbidden(protein, intervention)

        # Forbid intervention-intervention edges
        for int1 in int_cols:
            for int2 in int_cols:
                if int1 != int2:
                    obj.set_forbidden(int1, int2)
        
        return obj
    elif library == "causallearn":
        # create background object
        bk = BackgroundKnowledge()
        int_nodes = int_cols
        measured_nodes = measured_cols

        # forbid edges between intervention nodes
        for node1 in int_nodes:
            for node2 in int_nodes:
                if node1 != node2:
                    bk.add_forbidden_by_node(node1, node2)

        # forbid edges from measured to intervention variables
        for node in int_nodes:
            for measured_node in measured_nodes:
                bk.add_forbidden_by_node(measured_node, node)
        
        return bk
