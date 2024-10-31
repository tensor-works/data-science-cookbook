import plotly.graph_objects as go
import plotly.express as px


def plot_clusters(df, pc1_name, pc2_name, hue_column):
    """
    Plots a scatter plot of the PCA components with color coding based on the hue column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the PCA components and the hue column.
        pc1_name (str): The name of the column for the first principal component.
        pc2_name (str): The name of the column for the second principal component.
        hue_column (str): The column name to be used for color coding the points.
    
    Returns:
        None: Displays the plot.
    """

    # Create the figure
    fig = go.Figure()

    # Unique values for color coding and markers
    clusters = df[hue_column].unique()
    markers = ['circle', 'diamond', 'cross', 'square']
    colors = px.colors.qualitative.Plotly

    # Add traces for each cluster
    for i, cluster in enumerate(clusters):
        cluster_data = df[df[hue_column] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data[pc1_name],
                y=cluster_data[pc2_name],
                mode='markers',
                marker=dict(
                    size=5,
                    symbol=markers[i % len(markers)],  # Cycle through markers
                    color=colors[i % len(colors)],  # Cycle through colors
                    opacity=0.8),
                name=f'Cluster {cluster}'))

    # Update layout
    fig.update_layout(title='PCA Scatter Plot by Clusters',
                      xaxis_title=pc1_name,
                      yaxis_title=pc2_name,
                      legend_title=hue_column)

    # Show the plot
    fig.show()
