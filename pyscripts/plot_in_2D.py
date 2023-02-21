import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import umap
import os
import seaborn as sns

#load data
features = np.genfromtxt(snakemake.input['features'][0], delimiter = ',')
class_labels_pd = pd.read_csv(snakemake.input['class_labels'], header=None)
class_labels = class_labels_pd[0].tolist()

#create directories to save plots
if snakemake.params['scDINO_full_pipeline']:
    save_dir= f"{snakemake.wildcards['save_dir_downstream_run']}/{snakemake.wildcards['ViT_name']}_channel{snakemake.wildcards['channel_names']}_analyses/embedding_plots/"
    file_name= f"epoch{snakemake.wildcards['epoch_num']}_"
    umap_params = snakemake.config['downstream_analyses']['umap_eval']
else:
    save_dir= f"{snakemake.wildcards['save_dir_downstream_run']}/embedding_plots/"
    file_name= f"channel{snakemake.wildcards['channel_names']}_model_{snakemake.wildcards['model_name']}_"
    umap_params = snakemake.config['umap_eval']

#UMAP
def fit_umap(data, n_neighbors, min_dist, metric, spread, epochs):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, n_epochs=epochs, random_state=42)
    umap_embedding = umap_model.fit_transform(data)
    return umap_embedding

n_neighbors = umap_params['n_neighbors']
min_dist = umap_params['min_dist']
metric = umap_params['metric']
spread = umap_params['spread']
epochs = umap_params['epochs']
umap_embedding = fit_umap(features, n_neighbors=15, min_dist=0.1, metric=metric, spread=spread, epochs=epochs)

print(features.shape)
print(len(class_labels))
print(umap_embedding.shape)
# custom_palette = sns.color_palette("viridis", len(set(class_labels)))
# import colorcet as cc
# custom_palette = sns.color_palette(cc.glasbey, n_colors=len(set(class_labels)))
custom_palette = sns.color_palette("hls", len(set(class_labels)))

def make_plot(embedding, labels, save_dir, file_name=file_name,name="Emb type", description="details"):
    sns_plot = sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=14, palette=custom_palette, edgecolor='black', linewidth=0.05)
    plt.suptitle(f"{name}_{file_name}", fontsize=8)
    sns_plot.tick_params(labelbottom=False)
    sns_plot.tick_params(labelleft=False)
    sns_plot.tick_params(bottom=False)
    sns_plot.tick_params(left=False)
    sns_plot.set_title("CLS Token embedding of "+str(len(labels))+" cells with a dimensionality of "+str(features.shape[1])+" \n"+description, fontsize=6)
    sns.move_legend(sns_plot, "upper right", title='Classes', prop={'size': 5}, title_fontsize=6, markerscale=0.5)
    sns.set(rc={"figure.figsize":(14, 10)})
    sns.despine(bottom = True, left = True)
    sns_plot.figure.savefig(f"{save_dir}{file_name}{name}.png", dpi=200)
    sns_plot.figure.savefig(f"{save_dir}pdf_format/{file_name}{name}.pdf")
    plt.close()

os.makedirs(f"{save_dir}pdf_format", exist_ok=True)

make_plot(umap_embedding, class_labels, save_dir=save_dir, file_name=file_name, name="umap",description=f"n_neighbors:{n_neighbors}, min_dist={min_dist}, metric={metric}, spread={spread}, epochs={epochs}")


########################### Additional plots from https://topometry.readthedocs.io/en/latest/ ###########################
if snakemake.params['topometry_plots']:

    import topo as tp

    os.makedirs(f"{save_dir}/topometry_plots", exist_ok=True)
    os.makedirs(f"{save_dir}/topometry_plots/pdf_format", exist_ok=True)

    save_dir_topo = f"{save_dir}/topometry_plots/"
    # Learn topological metrics and basis from data. The default is to use diffusion harmonics.
    tg = tp.TopOGraph()

    print('running all combinations')
    tg.run_layouts(features, n_components=2,
                        bases=['diffusion', 'fuzzy'],
                        graphs=['diff', 'fuzzy'],
                        layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])

    make_plot(tg.db_diff_MAP, class_labels, name="db_diff_MAP", save_dir=save_dir_topo)
    make_plot(tg.db_fuzzy_MAP, class_labels, name="db_fuzzy_MAP", save_dir=save_dir_topo)
    make_plot(tg.db_diff_MDE, class_labels, name="db_diff_MDE", save_dir=save_dir_topo)
    make_plot(tg.db_fuzzy_MDE, class_labels, name="db_fuzzy_MDE", save_dir=save_dir_topo)
    make_plot(tg.db_PaCMAP, class_labels, name="db_PaCMAP", save_dir=save_dir_topo)
    make_plot(tg.db_TriMAP, class_labels, name="db_TriMAP", save_dir=save_dir_topo)
    make_plot(tg.db_tSNE, class_labels, name="db_tSNE", save_dir=save_dir_topo)
    make_plot(tg.fb_diff_MAP, class_labels, name="fb_diff_MAP", save_dir=save_dir_topo)
    make_plot(tg.fb_fuzzy_MAP, class_labels, name="fb_fuzzy_MAP", save_dir=save_dir_topo)
    make_plot(tg.fb_diff_MDE, class_labels, name="fb_diff_MDE", save_dir=save_dir_topo)
    make_plot(tg.fb_fuzzy_MDE, class_labels, name="fb_fuzzy_MDE", save_dir=save_dir_topo)
    make_plot(tg.fb_PaCMAP, class_labels, name="fb_PaCMAP", save_dir=save_dir_topo)
    make_plot(tg.fb_TriMAP, class_labels, name="fb_TriMAP", save_dir=save_dir_topo)
    make_plot(tg.fb_tSNE, class_labels, name="fb_tSNE", save_dir=save_dir_topo)
