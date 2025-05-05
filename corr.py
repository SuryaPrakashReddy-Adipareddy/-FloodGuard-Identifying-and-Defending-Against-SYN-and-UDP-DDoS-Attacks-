from ddos1 import *

def create_visualizations(df, output_dir, prefix=""):
    # Feature distributions plot
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns[:6]):
        plt.subplot(2, 3, i+1)
        sns.kdeplot(data=df, x=column, hue='Label', common_norm=False)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"{prefix}feature_distributions.png"))
    plt.close()

    # Enhanced correlation matrix
    plt.figure(figsize=(12, 10))
    
    # Select numeric columns and compute correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    
    # Sort correlation matrix by correlation strength
    ordered_corr = correlation.where(~np.tril(np.ones(correlation.shape)).astype(bool))
    ordered_corr = ordered_corr.abs().unstack()
    ordered_corr = ordered_corr.sort_values(ascending=False)
    ordered_cols = list(set([x[0] for x in ordered_corr.index if x[0] != x[1]]))
    
    # Reorder correlation matrix
    correlation = correlation.reindex(index=ordered_cols, columns=ordered_cols)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    # Plot correlation heatmap with enhanced styling
    sns.heatmap(correlation, 
                mask=mask,
                annot=True,  # Show correlation values
                fmt='.2f',   # Format to 2 decimal places
                cmap='RdYlBu_r',  # Red-Yellow-Blue diverging colormap
                center=0,    # Center colormap at 0
                square=True, # Make cells square
                linewidths=0.5,  # Add cell borders
                annot_kws={'size': 7},  # Annotation text size
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Feature Correlation Heatmap (Sorted by Correlation Strength)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "plots", f"{prefix}correlation_heatmap.png"), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

create_visualizations(train_df, output_dir, "train_")