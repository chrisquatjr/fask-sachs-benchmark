"""
Description: This script is designed to test various causal discovery algorithms on the Sachs dataset

Author: Chris Andersen
Date: 12/5/2024

Notes on statistics reported below:

TP : number of adjacencies/arrowheads shared by true graph (baseline_graph) and estimated graph (tet_fask, etc.)
FP : number of adjacencies/arrowheads in estimated graph that are not in true graph
FN : number of adjacencies/arrowheads in true graph that are not in estimated graph
TN : number of adjacencies/arrowheads not in true graph and not in estimated graph

--- Precision = TP / (TP + FP), Recall = TP / (TP + FN)
precision gives the proportion of adjacencies/arrowheads in the estimated graph that are correct while
recall gives the proportion of adjacencies/arrowheads in the true graph that are correctly estimated

"""
# basic imports
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import graphviz as gviz
import matplotlib.pyplot as plt
import seaborn as sns

# Start JVM
import jpype.imports
import importlib.resources as importlib_resources

jar_path = importlib_resources.files('pytetrad').joinpath('resources','tetrad-current.jar')
jar_path = str(jar_path)
if not jpype.isJVMStarted():
    try:
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[jar_path])
    except OSError:
        print("can't load jvm")
        pass

# import pytetrad functions
import pytetrad.tools.TetradSearch as ts

# import causal learn functions
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam

# Get the directory where the current script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Define project root as one level up from scripts directory
PROJECT_ROOT = SCRIPT_DIR.parent

# Add scripts directory to Python path for importing local modules
sys.path.append(str(SCRIPT_DIR))
import pytetrad_sachs_utilities as psu

# Define data directory paths relative to project root
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "results"
TSV_DIR = DATA_DIR / "tsvs"

visualize = True
int_index = 11

# check if log data file already exists
try:
    log_df = pd.read_csv(DATA_DIR / "data.txt", sep=",")
except FileNotFoundError:
    sachs_with_interv = pd.read_csv(TSV_DIR / "sachs_jittered_combined.tsv", sep="\t")
    log_df = sachs_with_interv.apply(lambda x: np.log2(x + 10))
    # save log transformed data
    log_df.to_csv(TSV_DIR / "sachs_jittered_combined_log2.tsv", sep="\t", index=False)

# organize data and save intervention column names
log_df = log_df.astype({col: "float64" for col in log_df.columns})
int_cols = list(log_df.columns[int_index:])
measured_cols = list(log_df.columns[:int_index])

# create mapping from node names to generic node naming convention
node_names = {f"X{i+1}": col for i, col in enumerate(log_df.columns)}
pattern = r'\d+\.\s+([\w_]+)\s+(-->|<->|o->|o-o)\s+([\w_]+)'

# visualize transformation (only for observed variables, not interventions)
if visualize:
    log_vis = log_df.iloc[:, :int_index]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(log_vis.values.flatten(), bins=100, color="blue")
    plt.title("Log2 Transformed Sachs Data (Observed Variables Only)")
    plt.xlabel("Log Expression Value")
    plt.ylabel("Frequency")
    plt.savefig(OUT_DIR / "sachs_jittered_combined_log2.png")
    plt.close()

    # visualize correlation
    corr = log_vis.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig(OUT_DIR / "sachs_jittered_combined_log2_corr.png")
    plt.close()

### create lists of known edges

# --- define ground truth dataframes ---
# gt_df = Sachs et al. 2005 biological model
# r_gt_df = Ramsey et al. 2018 "supplemental ground truth"
# r_fig_5_df = Ramsey et al. 2018 representation of Sachs' estimated model
# r_fask_df = Ramsey et al. 2018 FASK model
gt_df, r_gt_df, r_fig_5_df, r_fask_df = psu.baseline_models()

# convert one of the ground truth dataframes to a graph
baseline_graph, tetrad_data = psu.create_baseline_model(gt_df,measured_cols)
# create edge dataframe for the chosen ground truth
ground_truth_set = "sachs_biological"
edge_df = gt_df.copy()

# borrow figure 7 from Ramsey et al. 2018 for comparison
r_fask_graph, r_fask_tetrad_data = psu.create_baseline_model(r_fask_df,measured_cols)
r_fask_str = r_fask_graph.__str__()

r_fask_stats = psu.compute_statistics(baseline_graph, r_fask_str, tetrad_data, node_names)

stats_df = pd.DataFrame(columns=["adjacency_precision", "adjacency_recall", "arrowhead_precision", "arrowhead_recall"])
stats_df.loc["ramsey_fask"] = r_fask_stats.values() # target values I want to try to hit

# visualize baseline
gdot = gviz.Graph(format='png', 
                  engine='dot', 
                  graph_attr={'ratio': 'fill',           # fill available space
                             'size': '8,8',              # Sets the target size in inches
                             'margin': '0.5',            # Adds margin around the graph
                             'dpi': '300',               # Increases resolution
                             'splines': 'spline',        # Makes edges curve more naturally
                             'overlap': 'scale',         # Prevents node overlap by scaling
                             'outputorder': 'edgesfirst'})
psu.write_gdot(baseline_graph, gdot, node_mapping=node_names)
gdot.render(filename=OUT_DIR / f"{ground_truth_set}", cleanup=True, quiet=True)
gdot.clear()

### Run FASK using pytetrad using parameters described in this manuscript: https://arxiv.org/pdf/1805.03108

fask_search = ts.TetradSearch(log_df)

# run fask with background info
fask_search = psu.add_background_knowledge(int_cols=int_cols, measured_cols=measured_cols, obj=fask_search)
fask_search.use_sem_bic()
fask_search.run_fask(alpha=0.00001,
                     depth=-1,
                     fask_delta=-0.2,
                     left_right_rule=1,
                     skew_edge_threshold=0.3
                     )
fask_str = fask_search.__str__()
fask_str = psu.remove_edges(fask_str, int_cols)
with open(DATA_DIR / "fask_str.txt", "w") as text_file:
    text_file.write(fask_str)
edge_df = psu.add_edges_column(df=edge_df,
                           col="tetrad_FASK",
                           pattern=pattern,
                           out_str=fask_str)

# compute statistics
fask_stats = psu.compute_statistics(baseline_graph, fask_str, tetrad_data, node_names)
stats_df.loc["tetrad_FASK"] = fask_stats.values()

# visualize
filtered_dag = psu.create_filtered_graph(fask_str, measured_cols)
gdot = gviz.Graph(format='png', 
                  engine='dot', 
                  graph_attr={'ratio': 'fill',           # fill available space
                             'size': '8,8',              # Sets the target size in inches
                             'margin': '0.5',            # Adds margin around the graph
                             'dpi': '300',               # Increases resolution
                             'splines': 'spline',        # Makes edges curve more naturally
                             'overlap': 'scale',         # Prevents node overlap by scaling
                             'outputorder': 'edgesfirst'})
psu.write_gdot(filtered_dag, gdot, node_mapping=node_names)
gdot.render(filename=OUT_DIR / "pytetrad_FASK_graph", cleanup=True, quiet=True)
gdot.clear()

### Run FCI using pytetrad

fci_search = ts.TetradSearch(log_df)
fci_search = psu.add_background_knowledge(obj=fci_search, int_cols=int_cols, measured_cols=measured_cols)
fci_search.use_fisher_z(alpha=0.005)
fci_search.run_fci(depth=-1, 
                   stable_fas=True,
                   max_disc_path_length=-1,
                   complete_rule_set_used=True,
                   guarantee_pag=False
                   )
fci_str = fci_search.__str__()
fci_str = psu.remove_edges(fci_str, int_cols)
with open(DATA_DIR / "fci_str.txt", "w") as text_file:
    text_file.write(fci_str)
edge_df = psu.add_edges_column(edge_df, "tetrad_FCI", pattern, fci_str)

# compute statistics
tfci_stats = psu.compute_statistics(baseline_graph, fci_str, tetrad_data, node_names)
stats_df.loc["tetrad_FCI"] = tfci_stats.values()

# visualize
dag = psu.create_filtered_graph(fci_str, measured_cols)
gdot = gviz.Graph(format='png', 
                  engine='dot', 
                  graph_attr={'ratio': 'fill',           # fill available space
                             'size': '8,8',              # Sets the target size in inches
                             'margin': '0.5',            # Adds margin around the graph
                             'dpi': '300',               # Increases resolution
                             'splines': 'spline',        # Makes edges curve more naturally
                             'overlap': 'scale',         # Prevents node overlap by scaling
                             'outputorder': 'edgesfirst'})
psu.write_gdot(dag, gdot, node_mapping=node_names)
gdot.render(filename=OUT_DIR / "pytetrad_FCI_graph", cleanup=True, quiet=True)
gdot.clear()

### Run GES using pytetrad

ges_search = ts.TetradSearch(log_df)
ges_search = psu.add_background_knowledge(obj=ges_search, int_cols=int_cols, measured_cols=measured_cols)
ges_search.use_sem_bic()
ges_search.run_fges(symmetric_first_step=False, 
                    max_degree=-1, 
                    parallelized=False, 
                    faithfulness_assumed=False
                    )
ges_str = ges_search.__str__()
# Using two separate patterns - more flexible if the format might change
ges_score = float(re.search(r'Score: (-?\d+\.\d+)', ges_str).group(1))
ges_str = ges_str.partition('Graph Attributes:')[0]
ges_str = psu.remove_edges(ges_str, int_cols)
with open(DATA_DIR / "ges_str.txt", "w") as text_file:
    text_file.write(ges_str)
edge_df = psu.add_edges_column(edge_df, "tetrad_GES", pattern, ges_str)

# compute statistics
tges_stats = psu.compute_statistics(baseline_graph, ges_str, tetrad_data, node_names)
stats_df.loc["tetrad_GES"] = tges_stats.values()

# visualize
dag = psu.create_filtered_graph(ges_str, measured_cols)
gdot = gviz.Graph(format='png', 
                  engine='dot', 
                  graph_attr={'ratio': 'fill',           # fill available space
                             'size': '8,8',              # Sets the target size in inches
                             'margin': '0.5',            # Adds margin around the graph
                             'dpi': '300',               # Increases resolution
                             'splines': 'spline',        # Makes edges curve more naturally
                             'overlap': 'scale',         # Prevents node overlap by scaling
                             'outputorder': 'edgesfirst'})
psu.write_gdot(dag, gdot, node_mapping=node_names)
gdot.render(filename=OUT_DIR / "pytetrad_GES_graph", cleanup=True, quiet=True)
gdot.clear()

### Run FCI using causal-learn

# reformat as numpy array then run FCI
log_array = log_df.to_numpy()
g, edges = fci(dataset=log_array,
            independence_test_method="fisherz",
            alpha=0.005,
            node_names=None
            )
nodes = g.get_nodes()
int_nodes = nodes[int_index:]
measured_nodes = nodes[:int_index]

# create background knowledge
bk = psu.add_background_knowledge(int_cols=int_nodes, measured_cols=measured_nodes, library="causallearn")
g, edges = fci(dataset=log_array,
            independence_test_method="fisherz",
            alpha=0.005,
            node_names=None,
            background_knowledge=bk
            )
fci_str = psu.replace_nodes(g.__str__(), node_names)
fci_str = psu.remove_edges(fci_str, int_cols)
with open(DATA_DIR / "clearn_FCI_str.txt", "w") as text_file:
    text_file.write(fci_str)
edge_df = psu.add_edges_column(edge_df, "clearn_FCI", pattern, fci_str)

# compute statistics
cfci_stats = psu.compute_statistics(baseline_graph, fci_str, tetrad_data, node_names)
stats_df.loc["clearn_FCI"] = cfci_stats.values()

# create clearn graph output
pdy = GraphUtils.to_pydot(g,labels=log_df.columns)
pdy.write_png(OUT_DIR / "clearn_FCI_graph.png")

### Run GES using causal-learn
# Seemingly no way to incorporate background knowledge with this implementation, so we will run GES without it
meas_array = log_array[:,:int_index] # simply ignore intervention variables
Record = ges(X=meas_array,
             score_func="local_score_BIC", 
             maxP=None, 
             parameters=None,
             node_names=measured_cols     # This feature is implemented in this function, however...
             )
g = Record['G']; ges_score = Record['score']
ges_str = g.__str__()
edge_df = psu.add_edges_column(edge_df, "clearn_GES", pattern, ges_str)

# compute statistics
cges_stats = psu.compute_statistics(baseline_graph, ges_str, tetrad_data, node_names)
stats_df.loc["clearn_GES"] = cges_stats.values()

# create clearn graph output
pdy = GraphUtils.to_pydot(g)
pdy.write_png(OUT_DIR / "clearn_GES_graph.png")

### Run LiNGAM using causal-learn (since our data contains strong non-Gaussianities)
# also no obvious way to incorporate background knowledge, so we will run LiNGAM without it
model = lingam.ICALiNGAM()
model.fit(meas_array)

# Get graph string representation
lingam_str = psu.adjacency_to_graph_string(model.adjacency_matrix_, measured_cols)
edge_df = psu.add_edges_column(edge_df, "clearn_LiNGAM", pattern, lingam_str)

# Create visualization
cg = psu.adjacency_to_graph(model.adjacency_matrix_, measured_cols)
pdy = GraphUtils.to_pydot(cg.G, labels=measured_cols)
pdy.write_png(OUT_DIR / "clearn_LiNGAM_graph.png")

# Compute statistics
lingam_stats = psu.compute_statistics(baseline_graph, lingam_str, tetrad_data, node_names)
stats_df.loc["clearn_LiNGAM"] = lingam_stats.values()

# save final results
edge_df.to_csv(OUT_DIR / "computed_edges.tsv", sep="\t", index=False)
stats_df.to_csv(OUT_DIR / "stats.tsv", sep="\t", index=True)

# print out the statistics for each method tested
print(stats_df)
