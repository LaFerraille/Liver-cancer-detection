import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns


def visualize_slice_ranges(multislice_complete, limit=10):
    # Get unique IDs from the DataFrame
    unique_ids = multislice_complete['ID'].unique()

    # Initialize lists to store results and IDs with range > limit
    ranges = []
    slices_range = []
    ids_with_range_gt = []

    # Iterate over unique IDs
    for id in unique_ids:
        # Filter data once per patient
        patient_data = multislice_complete[multislice_complete['ID'] == id]
        min_slice = patient_data['slice_num'].min()
        max_slice = patient_data['slice_num'].max()
        
        # Calculate range and store results
        patient_range = max_slice - min_slice
        ranges.append(patient_range)
        slices_range.append((min_slice, max_slice))
        if patient_range >= limit:
            ids_with_range_gt.append(id)

    # Setup figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))  # 2 rows, 1 column

    # Plot the cumulative distribution of slice number ranges on the first subplot
    axs[0].hist(ranges, bins=50, cumulative=-1, edgecolor='black', histtype='step', color='blue')
    axs[0].set_xlabel("Slice Number Range")
    axs[0].set_ylabel("Complementary Cumulative Count of Patient IDs")
    axs[0].axhline(y=len(ids_with_range_gt), color='red', linestyle='--', linewidth=2)
    axs[0].text(x=max(ranges)*0.78, y=len(ids_with_range_gt)+1, s=f" {len(ids_with_range_gt)} patients (slice range > {limit})", verticalalignment='center', color='red')

    # Plot horizontal lines from min to max slice number for each patient on the second subplot
    for i, id in enumerate(unique_ids):
        scaled_index = i * 0.5  # Reduce the vertical spacing by scaling the index
        axs[1].hlines(scaled_index, slices_range[i][0], slices_range[i][1], color='blue')
        axs[1].text(slices_range[i][0]-4, scaled_index, str(slices_range[i][0]), color='blue', va='center')
        axs[1].text(slices_range[i][1], scaled_index, str(slices_range[i][1]), color='blue', va='center')

    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_ylim(-1, len(unique_ids) * 0.5)

    # Improve layout to avoid overlap and show the plot
    plt.tight_layout()
    plt.show()

def pca_factorial_plot(X, y):

    # Standardize the data
    X_scaled = StandardScaler().fit_transform(X)
    y = y.copy()
    y.reset_index(drop=True, inplace=True)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    #Create a DataFrame for easier plotting
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Group'] = y # Assuming 'y' is your categorical label

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Group', data=pca_df, palette='Set1', alpha=0.7)

    # Set the aspect of the plot to be equal, to keep PC1 and PC2 on the same scale
    plt.gca().set_aspect('equal', adjustable='datalim')
    #plt.title('PCA Factorial Plan')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')

    # Enhance the legend
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Draw a horizontal and vertical line passing through zero to better visualize the center
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.axvline(0, color='grey', linewidth=0.8)

    # Improve grid visibility and style
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(-100, 100)
    plt.ylim(-80, 80)

    plt.show()

def pca_loading_plot(X, top_features=-1, verbose=False):
    # Standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Calculate loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Create a DataFrame for easier manipulation
    loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=X.columns)

    # Calculate the magnitude of each vector
    loading_df['Magnitude'] = np.sqrt(loading_df['PC1']**2 + loading_df['PC2']**2)

    # Sort features by 'Magnitude' to get the most important ones
    loading_df = loading_df.sort_values(by='Magnitude', ascending=False)

    # How many top features to display
    top_features = top_features

    # Plot settings
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Plot unit circle
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)
    ax.add_artist(circle)

    # Plot only the most important features
    for i, feature in enumerate(loading_df.index[:top_features]):
        vector = loading_df.loc[feature]
        ax.arrow(0, 0, vector['PC1'], vector['PC2'], color='r', alpha=0.5, head_width=0.02, head_length=0.03)
        if verbose : 
            plt.text(vector['PC1'] * 1.1, vector['PC2'] * 1.1, feature, fontsize=10)  # Slight offset for label

    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
    #plt.title('PCA Loading Plot - Top Contributing Features')
    plt.grid(True)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)

    # Set limits to ensure the unit circle is completely visible
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure circle is round
    plt.show()

def pca_scree_plot(X):
    # Standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Display the summary
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot settings
    plt.figure(figsize=(12, 8))
    individual_line, = plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, 'o--', label='Individual Explained Variance')
    cumulative_line, = plt.plot(np.arange(1, len(explained_variance) + 1), cumulative_variance, 'o-', color='r', label='Cumulative Explained Variance')

    #plt.title('PCA Scree Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained (%)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(handles=[individual_line, cumulative_line])

    # Adjusting x-axis ticks to only show those components explaining at least 5% of variance
    significant_components = np.linspace(1,len(explained_variance),5) #[i + 1 for i, v in enumerate(explained_variance) if v >= 0.05]
    plt.xticks(significant_components)

    # Annotate each significant point
    for i, v in enumerate(explained_variance):
        if v >= 0.05:  # only annotate significant components
            label = f"{v:.2f}"
            plt.annotate(label, (i + 1, v), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

def compute_several_plot(df):

    palette = {'CCK': 'tab:red', 'CHC': 'tab:blue', 'Mixtes': 'tab:green'}

    plt.figure(figsize=(20, 10))
    plt.subplot(4, 4, 1)
    sns.histplot(data=df, x='Age_at_disease', hue='classe_name', multiple='stack', palette=palette, bins=20)

    subtype_counts = df['classe_name'].value_counts()
    explode = [0.1 if i == subtype_counts.idxmax() else 0 for i in subtype_counts.index]  # Explode the largest segment
    colors = [palette[subtype] for subtype in subtype_counts.index]  # Assign colors consistently
    plt.subplot(4, 4, 2)
    subtype_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=explode, colors=colors)
    plt.axis('off') 

    plt.subplot(4, 4, 3)
    df['Survived'] = 1 - df['Death']
    survival_rates = df.groupby(['classe_name', 'Gender']).agg({
        'Survived': 'mean'
    }).reset_index()
    survival_rates['Survival Rate (%)'] = survival_rates['Survived'] * 100
    sns.barplot(data=survival_rates, x='classe_name', y='Survival Rate (%)', hue='Gender', palette='Set2')
    #plt.title('Percentage of Survivors in Each Cancer Subtype by Gender')
    plt.ylabel('Survival Rate (%)')
    plt.xlabel('Cancer Subtype')
    plt.legend(title='Gender')

    plt.subplot(4, 4, 4)
    # Alpha-fetoprotein Levels by Carcinoma Sub-Type
    sns.boxplot(data=df, x='classe_name', y='Alpha_foetoprotein', palette=palette)
    #plt.title('Alpha-fetoprotein Levels by Liver Cancer Subtype')
    plt.ylabel('Alpha-fetoprotein Level')
    plt.xlabel('Cancer Subtype')

    # Gender Distribution
    plt.subplot(4, 4, 5)
    sns.countplot(x='Gender', data=df)
    plt.title('Gender Distribution')

    # Age Distribution
    plt.subplot(4, 4, 6)
    sns.histplot(data=df, x='Age_at_disease', bins=20)
    plt.title('Age Distribution')

    # Alpha-fetoprotein Distribution
    plt.subplot(4, 4, 7)
    sns.boxplot(x='classe_name', y='Alpha_foetoprotein', data=df)
    plt.title('Alpha-fetoprotein Distribution')

    # Survival Status
    plt.subplot(4, 4, 8)
    death_counts = df['Death'].value_counts()
    plt.pie(death_counts, labels=death_counts.index.map({0: 'Alive', 1: 'Died'}), autopct='%1.1f%%')
    plt.title('Survival Status')


    plt.tight_layout()
    plt.show()

def plot_correl(df):
    # Reorganize df
    for subtype in df['classe_name'].unique():
        df[subtype] = (df['classe_name'] == subtype).astype(int)

    df['Survived'] = 1 - df['Death']
    df['gender'] = (df['Gender'] == 'M').astype(int)
    df = df.dropna(subset=['Survived'])
    df = df.dropna(subset=['Local_relapse'])
    df = df.dropna(subset=['Distant_relapse'])
    df = df.dropna(subset=['Alpha_foetoprotein'])
    df = df[df['Alpha_foetoprotein'] != df['Alpha_foetoprotein'].max()]
    feature_columns = ['Age_at_disease', 'Alpha_foetoprotein', 'gender', 'Survived', 'Local_relapse', 'Distant_relapse']

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharex=True)

    for i, subtype in enumerate(df['classe_name'].unique()):
        correlations = df[feature_columns].corrwith(df[subtype]).sort_values(ascending=False)
        
        # Plot the correlations
        sns.barplot(ax=axes[i], x=correlations.values, y=correlations.index, palette='coolwarm')
        axes[i].set_title(f'Correlation with {subtype} Subtype Presence')

    plt.tight_layout()
    plt.show()

def plot_phase_evolution(df, selected_feature=['original_glcm_ClusterTendency']):

    df = df.reset_index() 
    global_radio_complete = df.copy()

    global_radio_complete_reset = global_radio_complete.reset_index()
    palette = {'CCK': 'tab:red', 'CHC': 'tab:blue', 'Mixtes': 'tab:green'}

    for feat in selected_feature:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, phase in enumerate(global_radio_complete_reset['temps_inj'].unique()):
            phase_data = global_radio_complete_reset[global_radio_complete_reset['temps_inj'] == phase]
            if not phase_data.empty:
                sns.kdeplot(data=phase_data, x=feat, hue='classe_name', fill=True,
                            common_norm=False, ax=axes[i], palette=palette)
                
                handles, labels = axes[i].get_legend_handles_labels()
                if labels:
                    axes[i].legend(handles, labels, title='Subtype')

                axes[i].set_title(f'{phase} Phase - {feat.split("_")[-1]}')
                axes[i].set_xlabel(feat.split("_")[-1])
                axes[i].set_ylabel('Density')

        plt.suptitle(f'Density Plot of {feat.split("_")[-1]} by Subtype and Phase', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


