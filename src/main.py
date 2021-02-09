import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn.datasets as datasets
import sklearn.manifold as manifold

if os.path.exists("../image") is False:
    os.mkdir("../image")

if __name__ == "__main__":
    # Data Loading from sklearn.datasets
    data = datasets.fetch_openml(
        'mnist_784',
        version = 1,
        return_X_y = True
    )
    pixel_values, targets = data
    targets = targets.astype(int)

    # Save an example image of the dataset
    single_image = pixel_values.iloc[0].values.reshape(28,28)
    plt.imsave("../image/single_image.png", single_image, cmap="gray")

    # Create the t-SNE transformation of the data
    tsne = manifold.TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(pixel_values.iloc[:3000])
     
    # The data need to be converted to pandas dataframe, moreover the 
    # two dimensional components are stacked together with the corresponding 
    # target
    tsne_df = pd.DataFrame(
        np.column_stack((transformed_data, targets[:3000])),
        columns = ["x", "y", "targets"]
        )
    tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int) 

    # Visualization of the images of the dataset as points in a two dimensional
    # space
    grid = sns.FacetGrid(tsne_df, hue = "targets", height = 10)
    grid.map(plt.scatter, "x", "y").add_legend().savefig("../image/output.png")
    