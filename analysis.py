import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os


def get_unique_top_words(phi, topic_nums, top_num, metaname, thr=0.1):
    """
    Extract unique top words from topics.

    Parameters:
    -----------
    phi : numpy.ndarray
        Topic-word distribution matrix.
    topic_nums : list
        List of topic numbers to consider.
    top_num : int
        Number of top words to consider per topic.
    metaname : list
        List of word names/tokens.
    thr : float, optional
        Threshold for word probability, by default 0.1.

    Returns:
    --------
    mat : numpy.ndarray
        Matrix of probabilities for unique top words across topics.
    tokens : list
        List of corresponding token names.
    """
    for i in topic_nums:
        phi[i] = phi[i] / np.sum(phi[i])

    unique_word_indices = []
    appeared = set()

    for i in topic_nums:
        top_idx = np.argsort(phi[i])[::-1][:top_num]
        for idx in top_idx:
            if idx not in appeared and phi[i][idx] > thr:
                unique_word_indices.append(idx)
                appeared.add(idx)

    mat = []
    for i in topic_nums:
        mat.append(phi[i][unique_word_indices])
    mat = np.array(mat)

    tokens = [metaname[i] for i in unique_word_indices]
    return mat, tokens


def get_names_in_graph(csv_path, column='category'):
    """
    Read a CSV file and extract categories that have a non-empty index_in_graph value.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file.
    column : str, optional
        Column name to extract values from, by default 'category'.

    Returns:
    --------
    list
        List of values from the specified column for rows with valid index_in_graph.
    """
    categories_df = pd.read_csv(csv_path)
    filtered_categories = []

    for i in range(len(categories_df)):
        try:
            int_value = int(categories_df['index_in_graph'][i])
            filtered_categories.append(categories_df[column][i])
        except (ValueError, TypeError):
            continue

    return filtered_categories


def plot_umap(alpha, rho, csv_path, save_path, n_neighbors=30, min_dist=0.1, metric='cosine',
              custom_palette=None, title='UMAP Heatmap', label_topics=False):
    """
    Standardize, perform UMAP, and plot heatmap for alpha, rho, and category from CSV with topic labels.

    Parameters:
    -----------
    alpha : numpy.ndarray
        Alpha embedding matrix.
    rho : numpy.ndarray
        Rho embedding matrix.
    csv_path : str
        Path to CSV file containing category information.
    save_path : str
        Path to save the output figure.
    n_neighbors : int, optional
        UMAP parameter, by default 30.
    min_dist : float, optional
        UMAP parameter, by default 0.1.
    metric : str, optional
        UMAP distance metric, by default 'cosine'.
    custom_palette : dict, optional
        Custom color palette for categories, by default None.
    title : str, optional
        Plot title, by default 'UMAP Heatmap'.
    label_topics : bool, optional
        Whether to label topic points, by default False.
    """
    os.makedirs(save_path, exist_ok=True)
    scaler = StandardScaler()
    alpha_scaled = scaler.fit_transform(alpha)
    rho_scaled = scaler.fit_transform(rho)
    categories = pd.read_csv(csv_path)

    if 'category' not in categories.columns or 'index_in_graph' not in categories.columns:
        raise ValueError("CSV file must contain 'category' and 'index_in_graph' columns.")
    category_dict = dict(zip(categories['index_in_graph'], categories['category']))
    rho_categories = [category_dict.get(idx, 'Unknown') for idx in range(len(rho_scaled))]
    print(len(rho_scaled))

    combined_data = np.vstack([alpha_scaled, rho_scaled])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric)
    embeddings = reducer.fit_transform(combined_data)

    split_index = alpha_scaled.shape[0]
    embeddings_df = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'])
    embeddings_df['Type'] = ['Alpha'] * split_index + ['Rho'] * (embeddings.shape[0] - split_index)
    embeddings_df['Category'] = ['None'] * split_index + rho_categories

    topic_nums = list(range(split_index))
    embeddings_df.loc[embeddings_df['Type'] == 'Alpha', 'Topic'] = [f'Topic {i}' for i in topic_nums]

    plt.figure(figsize=(12, 10))
    auto_palette = None
    if custom_palette:
        sns.scatterplot(
            data=embeddings_df[embeddings_df['Type'] == 'Rho'],
            x='UMAP1', y='UMAP2', hue='Category', palette=custom_palette,
            marker='o', s=50)
    else:
        unique_categories = embeddings_df['Category'].unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_categories))
        sns.scatterplot(
            data=embeddings_df[embeddings_df['Type'] == 'Rho'],
            x='UMAP1', y='UMAP2', hue='Category', palette=palette,
            marker='o', s=50)
        auto_palette = {cat: palette[i] for i, cat in enumerate(unique_categories)}

    alpha_points = embeddings_df[embeddings_df['Type'] == 'Alpha']
    plt.scatter(alpha_points['UMAP1'], alpha_points['UMAP2'], color='black', marker='*', s=200)

    if label_topics:
        for idx, row in alpha_points.iterrows():
            plt.annotate(
                f'Topic {idx}',
                xy=(row['UMAP1'], row['UMAP2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7)
            )

    plt.title(title)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title='Data Type & Category', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    suffix = '_labeled' if label_topics else ''
    plt.savefig(f"{save_path}{title}{suffix}.png", bbox_inches='tight', dpi=300)
    plt.close()

    return auto_palette

def plot_topics1(phi, topic_nums, metaname, catname, top_num, catlut, tag, save_path, title,
                 thr=0.1, vmax=1, cmap="OrRd"):
    """
    Plot topic-word distributions with category information.
    Auto-scales based on number of topics and top features.

    Parameters:
    -----------
    phi : numpy.ndarray
        Topic-word distribution matrix.
    topic_nums : list
        List of topic numbers to visualize.
    metaname : list
        List of word names/tokens.
    catname : list
        List of category names for each word.
    top_num : int
        Number of top words to consider per topic.
    catlut : dict
        Lookup table mapping category names to colors.
    tag : str
        Label for y-axis.
    save_path : str
        Path to save the output figure.
    title : str
        Title for the figure.
    thr : float, optional
        Threshold for word probability, by default 0.1.
    vmax : float, optional
        Maximum value for color scaling, by default 1.
    cmap : str, optional
        Colormap name, by default "OrRd".
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.patches as mpatches

    # Get unique top words for topics
    m, tokens = get_unique_top_words(phi, topic_nums, top_num, metaname, thr)
    _, cat = get_unique_top_words(phi, topic_nums, top_num, catname, thr)

    m = m.T  # Transpose for visualization

    # Process categories for coloring
    unique_cat = list(sorted(np.unique(cat)))
    cat_index = {}
    for i, c in enumerate(unique_cat):
        cat_index[c] = i

    y = []
    for i in cat:
        if pd.isna(i):
            y.append(cat_index['nan'])
        else:
            y.append(cat_index[i])


    cmap1 = colors.ListedColormap([catlut.get(i, 'gray') for i in unique_cat])

    nwords, ntopics = m.shape

    # Dynamically set figure size based on number of topics and words
    # Scale height based on number of words
    height = max(10, nwords * 0.4)

    # Process words - conditionally remove substring and limit length
    word_list = [str(w) for w in tokens]
    processed_words = []
    for w in word_list:
        # Remove substring before first space only if ICD is in title
        if ' ' in w and 'icd' in title.lower():
            w = w[w.index(' ') + 1:]
        # Truncate long words
        if len(w) > 30:
            w = w[:30] + "..."
        processed_words.append(w)

    # Calculate suitable font size based on number of words
    word_fontsize = max(8, min(24, 500 / nwords))
    topic_fontsize = max(8, min(24, 500 / ntopics))

    # Create the figure with better proportions
    # Calculate proper width based on content (much less stretched)
    fig_width = max(12, ntopics * 0.4) + 5  # Reduce the base width calculation
    fig = plt.figure(figsize=(fig_width, height))

    # Create main axes for heatmap (left part) - make it less wide
    ax_heatmap = fig.add_axes([0.15, 0.1, 0.6, 0.8])  # [left, bottom, width, height]

    # Create small axes for category colors
    category_width = 0.02
    ax_category = fig.add_axes([0.76, 0.1, category_width, 0.8])

    # Create separate legend axes with proper spacing for text
    # Move it even further right and make it wider
    ax_legend = fig.add_axes([0.90, 0.1, 0.5, 0.8])  # Increased width significantly
    ax_legend.axis('off')  # Hide the axes

    # Create axes for colorbar (top)
    ax_colorbar = fig.add_axes([0.3, 0.92, 0.3, 0.02])

    # Plot heatmap in main axes
    im = ax_heatmap.imshow(m, aspect='auto', cmap=cmap, vmax=vmax, interpolation="none")

    # Plot categories in category axes if available with proper aspect ratio
    if not y:
        ax_category.axis("off")
    else:
        y = np.array(y)[:, np.newaxis]
        # Set the extent to match the heatmap grid
        extent = [-0.5, 0.5, -0.5 + nwords, -0.5]
        ax_category.imshow(y, cmap=cmap1, aspect='auto', extent=extent)
        ax_category.set_ylim(ax_heatmap.get_ylim())

    # Set labels in main axes
    ax_heatmap.set_yticks(np.arange(nwords), minor=False)
    ax_heatmap.set_yticklabels(processed_words, fontsize=word_fontsize, fontdict=None, minor=False)

    ax_heatmap.set_xticks(np.arange(ntopics), minor=False)
    ax_heatmap.set_xticklabels(topic_nums, fontdict=None, minor=False, rotation=45 if ntopics > 15 else 0,
                               fontsize=topic_fontsize)

    # Set category axes properties
    ax_category.set_xticks([])
    ax_category.set_yticks([])

    # Create patches for legend
    patches = []
    for c in unique_cat:
        patches.append(mpatches.Patch(color=catlut.get(c, 'gray'), label=c.lower()))

    # Place legend with more space between items
    legend = ax_legend.legend(handles=patches,
                              loc='center left',  # Align left
                              fontsize=max(8, min(18, 250 / len(unique_cat))),
                              frameon=True,
                              labelspacing=1.5,  # Increase vertical spacing between legend items
                              handletextpad=1.0)  # Increase spacing between handles and text

    # Set titles with adaptive font size
    title_fontsize = max(16, min(32, 800 / (ntopics + nwords)))
    ax_heatmap.set_xlabel('Topics', fontsize=title_fontsize)
    ax_heatmap.set_ylabel(tag, fontsize=title_fontsize)

    # Create colorbar in dedicated colorbar axes
    cb = fig.colorbar(im, cax=ax_colorbar, orientation='horizontal')
    cb.ax.tick_params(labelsize=max(8, min(16, 300 / ntopics)))

    plt.savefig(save_path + title + '.png', bbox_inches='tight', dpi=150)
    plt.close()


def plot_topics2(phi, topic_nums, metaname, catname, top_num, catlut, tag, save_path, title,
                 thr=0.1, figsize=(15, 15), bbox_to_anchor=(1., 0.25, 2, 2), vmax=1,
                 wratio=0.02, wspace=-0.88, cax=[0.6, 0.75, 0.3, 0.02], aspect=2, cmap="OrRd"):
    """
    Plot topic-word distributions with category information in a more customizable format.

    Parameters:
    -----------
    phi : numpy.ndarray
        Topic-word distribution matrix.
    topic_nums : list
        List of topic numbers to visualize.
    metaname : list
        List of word names/tokens.
    catname : list
        List of category names for each word.
    top_num : int
        Number of top words to consider per topic.
    catlut : dict
        Lookup table mapping category names to colors.
    tag : str
        Label for y-axis.
    save_path : str
        Path to save the output figure.
    title : str
        Title for the figure.
    thr : float, optional
        Threshold for word probability, by default 0.1.
    figsize : tuple, optional
        Figure size, by default (15, 15).
    bbox_to_anchor : tuple, optional
        Position for the legend, by default (1.,0.25, 2, 2).
    vmax : float, optional
        Maximum value for color scaling, by default 1.
    wratio : float, optional
        Width ratio, by default 0.02.
    wspace : float, optional
        Width space, by default -0.88.
    cax : list, optional
        Position for the color bar, by default [0.6, 0.75, 0.3, 0.02].
    aspect : int, optional
        Aspect ratio, by default 2.
    cmap : str, optional
        Colormap name, by default "OrRd".
    """
    os.makedirs(save_path, exist_ok=True)
    m, tokens = get_unique_top_words(phi, topic_nums, top_num, metaname, thr)
    _, cat = get_unique_top_words(phi, topic_nums, top_num, catname, thr)

    m = m.T

    unique_cat = list(sorted(np.unique(cat)))
    cat_index = {}
    for i, c in enumerate(unique_cat):
        cat_index[c] = i
    y = [cat_index[c] for c in cat]

    if catlut:
        cmap1 = colors.ListedColormap([catlut[c] for c in unique_cat])
    else:
        cmap1 = plt.cm.get_cmap('tab20', len(unique_cat))

    nwords, ntopics = m.shape

    grid = dict(height_ratios=[0.5, m.shape[0]], width_ratios=[wratio * m.shape[1], 1])
    fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw=grid, figsize=figsize)

    im = axes[1, 0].imshow(m, aspect='auto', cmap=cmap, vmax=vmax, interpolation="none")

    if not y:
        axes[1, 1].axis("off")
    else:
        y = np.array(y)[:, np.newaxis]
        axes[1, 1].imshow(y, cmap=cmap1)
        axes[1, 1].set_aspect(aspect)

    axes[0, 1].axis("off")

    # Process words - limit length
    word_list = [str(w) for w in tokens]
    word_list = [w[:30] + ("..." if len(w) > 30 else "") for w in word_list]

    axes[1, 0].set_yticks(np.arange(nwords), minor=False)
    axes[1, 0].set_yticklabels(word_list, fontsize=24, fontdict=None, minor=False)
    axes[1, 0].set_xticks(np.arange(ntopics), minor=False)
    axes[1, 0].set_xticklabels(topic_nums, fontdict=None, minor=False, rotation=0, fontsize=24)
    axes[1, 1].set_xticks([])

    patches = []
    for c in catlut.keys() if catlut else []:
        patches.append(mpatches.Patch(color=catlut[c], label=c.lower()))

    if patches:
        axes[1, 1].legend(handles=patches, loc='lower left', bbox_to_anchor=bbox_to_anchor, fontsize=24)

    axes[1, 0].set_xlabel('Topics', fontsize=32)
    axes[1, 0].set_ylabel(tag, fontsize=32)
    for ax in [axes[1, 1]]:
        ax.set_yticks([])
    plt.subplots_adjust(wspace=wspace, left=0.3, right=1.5)
    cax_axes = plt.axes(cax)
    cb = fig.colorbar(im, cax=cax_axes, orientation='horizontal', aspect=5)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(f"{save_path}{title}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_all_topics_in_batches(phi, vocab, cat, palette, topic_type, data_dir, batch_size=5, top_n=5, thr=0.00,
                               vmax=0.4, aspect=2):
    """
    Plot topics in batches, saving each batch as a separate figure.

    Parameters:
    -----------
    phi : array-like
        Topic-word distribution matrix
    vocab : list
        Vocabulary list
    cat : list
        Category list for each word
    palette : dict
        Color palette for categories
    topic_type : str
        Type of topics (e.g., "Conditions", "ICD10")
    data_dir : str
        Directory to save plots
    batch_size : int
        Number of topics to plot in each batch
    top_n : int
        Number of top words to show for each topic
    thr : float
        Threshold for word probabilities
    vmax : float
        Maximum value for color scale
    aspect : float
        Aspect ratio for the plot
    """
    total_topics = phi.shape[0]

    save_path = os.path.join(data_dir, 'plots/')
    os.makedirs(save_path, exist_ok=True)

    # Loop through topics in batches
    for start_idx in range(0, total_topics, batch_size):
        # Get the current batch of topics
        end_idx = min(start_idx + batch_size, total_topics)
        topic_batch = list(range(start_idx, end_idx))
        batch_title = f'Top_{topic_type}_per_Choosen_Topic{start_idx}_to_{end_idx - 1}'

        plot_topics2(
            phi,
            topic_batch,
            vocab,
            cat,
            top_n,
            palette,
            topic_type,
            save_path,
            batch_title,
            thr=thr,
            bbox_to_anchor=(1, 0, 2, 2),
            vmax=vmax,
            aspect=aspect
        )

        print(f"Plotted and saved topics {start_idx} to {end_idx - 1}")


if __name__ == '__main__':
    cond_palette = {
        'cardiovascular': '#8bcbfb',
        'haematology/dermatology': '#647be4',
        'endocrine/diabetes': '#f89393',
        'gastrointestinal/abdominal': '#990d59',
        'immunological/systemic disorders': '#992989',
        'infections': '#554b0b',
        'musculoskeletal/trauma': '#51f36e',
        'neurology/eye/psychiatry': '#d63939',
        'renal/urology': '#67a643',
        'gynaecology/breast': '#26690b',
        'respiratory/ent': '#e1cc4d',
        'various': '#ac6107',
        'None': 'black'
    }

    icd_palette = {
        'Chapter III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': '#FF6347',
        'Chapter XIV Diseases of the genitourinary system': '#4682B4',
        'Chapter XXII Codes for special purposes': '#D2B48C',
        'Chapter V Mental and behavioural disorders': '#DA70D6',
        'Chapter XXI Factors influencing health status and contact with health services': '#8A2BE2',
        'Chapter XVII Congenital malformations, deformations and chromosomal abnormalities': '#DC143C',
        'Chapter VI Diseases of the nervous system': '#FFD700',
        'Chapter IX Diseases of the circulatory system': '#3CB371',
        'Chapter I Certain infectious and parasitic diseases': '#FF4500',
        'Chapter XIII Diseases of the musculoskeletal system and connective tissue': '#2E8B57',
        'Chapter XIX Injury, poisoning and certain other consequences of external causes': '#B8860B',
        'Chapter IV Endocrine, nutritional and metabolic diseases': '#FFA500',
        'Chapter XI Diseases of the digestive system': '#8FBC8F',
        'Chapter XX External causes of morbidity and mortality': '#483D8B',
        'Chapter VIII Diseases of the ear and mastoid process': '#00BFFF',
        'Chapter XII Diseases of the skin and subcutaneous tissue': '#FF69B4',
        'Chapter II Neoplasms': '#A52A2A',
        'Chapter XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': '#778899',
        'Chapter X Diseases of the respiratory system': '#0000FF',
        'Chapter XV Pregnancy, childbirth and the puerperium': '#DB7093',
        'Chapter VII Diseases of the eye and adnexa': '#800080',
        'Chapter XVI Certain conditions originating in the perinatal period': '#FFB6C1',
        'None': 'black'
    }

    atc_palette = {
        'Unknown': 'black',
        'N': '#aec7e8',
        'R': '#ff7f0e',
        'C': '#ffbb78',
        'D': '#2ca02c',
        'A': '#98df8a',
        'S': '#d62728',
        'M': '#ff9896',
        'G': '#9467bd',
        'B': '#c5b0d5',
        'J': '#8c5646',
        'H': '#c49c94',
        'P': '#e377c2',
        'V': '#f7b6d2',
        'L': '#7f7f7f'
    }

    data_dir = './results/Mar20_Topic_30_CMSKP/'

    # Load embeddings and distributions
    alpha1 = np.load(f"{data_dir}alpha1.npy")
    alpha2 = np.load(f"{data_dir}alpha2.npy")
    alpha3 = np.load(f"{data_dir}alpha3.npy")

    rho1 = np.load(f"{data_dir}rho1.npy")
    rho2 = np.load(f"{data_dir}rho2.npy")
    rho3 = np.load(f"{data_dir}rho3.npy")

    phi1 = np.load(f"{data_dir}beta1.npy")
    phi2 = np.load(f"{data_dir}beta2.npy")
    phi3 = np.load(f"{data_dir}beta3.npy")

    # Load pre-trained embeddings
    cond_emb = np.load(f"./data/input/cond_emb_256.npy")
    icd_emb = np.load(f"./data/input/icd_emb_256.npy")
    med_node2vec_emb = torch.load(f"./data/atc_node_embeddings_256.pt").detach().numpy()

    cond_csv_path = "./data/cond_dict.csv"
    icd_csv_path = "./data/icd_dict.csv"
    med_csv_path = "./data/med_dict.csv"

    cond_cat = get_names_in_graph(cond_csv_path)
    icd_cat = get_names_in_graph(icd_csv_path)
    med_cat = get_names_in_graph(med_csv_path)

    cond_vocab = get_names_in_graph(cond_csv_path, column='meaning')
    icd_vocab = get_names_in_graph(icd_csv_path, column='meaning')
    med_vocab = get_names_in_graph(med_csv_path, column='meaning')

    plot_umap(alpha1, cond_emb, cond_csv_path, save_path=f"{data_dir}plots/", n_neighbors=15,
              title="Pre_Topic_30_Cond_UMAP_256", custom_palette=cond_palette, label_topics=True)
    plot_umap(alpha2, icd_emb, icd_csv_path, save_path=f"{data_dir}plots/", n_neighbors=15,
              title="Pre_Topic_30_ICD_UMAP_256", custom_palette=icd_palette, label_topics=True)
    plot_umap(alpha3, med_node2vec_emb, med_csv_path, save_path=f"{data_dir}plots/", n_neighbors=15,
              title='Pre_Topic_30_Med_UMAP_256', custom_palette=atc_palette, label_topics=True)

    plot_umap(alpha1, rho1, cond_csv_path, save_path=f"{data_dir}plots/", n_neighbors=15,
              title="Topic_30_Cond_UMAP_256", custom_palette=cond_palette, label_topics=True)
    plot_umap(alpha2, rho2, icd_csv_path, save_path=f"{data_dir}plots/", n_neighbors=15,
              title="Topic_30_ICD_UMAP_256", custom_palette=icd_palette, label_topics=True)
    plot_umap(alpha3, rho3, med_csv_path, save_path=f"{data_dir}plots/", n_neighbors=15,
              title='Topic_30_Med_UMAP_256', custom_palette=atc_palette, label_topics=True)


    cond_topics = range(30)
    plot_topics1(phi1, cond_topics, cond_vocab, cond_cat, 5, cond_palette, "Conditions",
                 save_path=f'{data_dir}/plots/', title='Top_Conditions_All_Topic',
                 thr=0.00, vmax=0.4)

    icd_topics = range(30)
    plot_topics1(phi2, icd_topics, icd_vocab, icd_cat, 5, icd_palette, "ICD10",
                 save_path=f'{data_dir}/plots/', title='Top_ICD10_Codes_All_Topic',
                 thr=0.00, vmax=0.4)

    med_topics = range(30)
    plot_topics1(phi3, med_topics, med_vocab, med_cat, 5, atc_palette, "Med",
                 save_path=f'{data_dir}/plots/', title='Top_Medication_All_Topic',
                 thr=0.00, vmax=0.4)

    # plot_all_topics_in_batches(
    #     phi=phi1,
    #     vocab=cond_vocab,
    #     cat=cond_cat,
    #     palette=cond_palette,
    #     topic_type="Conditions",
    #     data_dir=data_dir
    # )
    #
    # plot_all_topics_in_batches(
    #     phi=phi2,
    #     vocab=icd_vocab,
    #     cat=icd_cat,
    #     palette=icd_palette,
    #     topic_type="ICD10",
    #     data_dir=data_dir
    # )