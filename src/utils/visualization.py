import os
import glob
import imageio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings('ignore')

colors = list(mcolors.CSS4_COLORS)
moscow = gpd.read_file('ao-shape.zip')


def save_map(paths, terminals, day):
    fig, ax = plt.subplots(figsize=(16, 16))
    moscow.plot(ax=ax)

    for path, color in zip(paths, colors):
        ax.plot(terminals['longitude'].iloc[path], terminals['latitude'].iloc[path], c=color)

    plt.axis('off')
    plt.savefig(f'./assets/{day}.png')


def create_gif():
    with imageio.get_writer('routes.gif', mode='I') as writer:
        for filename in glob.glob('./assets/*.png'):
            image = imageio.imread(filename)
            writer.append_data(image)
