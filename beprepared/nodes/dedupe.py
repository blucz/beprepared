from beprepared.node import Node
from beprepared.workspace import Abort
from beprepared.dataset import Dataset
from beprepared.properties import CachedProperty

import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex
from pathlib import Path
import hashlib

class ExactDedupe(Node):
    '''Deduplicates images based on their SHA256 hash. This is extremely fast and simple, but does not catch perceptually similar images. If that is a priority, try `FuzzyDedupe` instead, or better yet
    use both in sequence.'''
    def __init__(self):
        '''Create a new ExactDedupe node.'''
        super().__init__()

    def eval(self, dataset: Dataset) -> Dataset:
        visited = set()
        prev_count = len(dataset.images)
        dataset.images = [image for image in dataset.images 
                            if image.objectid.value not in visited and not visited.add(image.objectid.value)]
        self.log.info(f"Removed {prev_count - len(dataset.images)} duplicates from dataset ({100 * (prev_count - len(dataset.images)) / prev_count:.1f}%)")
        return dataset

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px

class FuzzyDedupe(Node):
    '''Deduplicates images based on perceptual similarity. 

    This process uses clip embeddings and an ANN (approximate nearest neighbors) index to find groups of images that are within `threshold` cosine similarity of each other. You can monitor the clusters by setting `debug_html` to a path where an HTML file will be saved with the images in each cluster.

    The n_trees and n_neighbors parameters control the accuracy and speed of the ANN index. Higher values will be more accurate but slower. The default values are usually good enough for most cases.
    '''
    def __init__(self, threshold: float = 0.95, debug_html='fuzzy_dedupe.html', n_trees: int = 10, n_neighbors: int = 50):
        '''Create a new FuzzyDedupe node.

        Args:
        - threshold (float): The cosine similarity threshold for images to be considered duplicates. Default is 0.95.
        - debug_html (str): If set, an HTML file will be saved with the images in each cluster for quality monitoring. Default is 'fuzzy_dedupe.html'.
        - n_trees (int): The number of trees to build in the Annoy index. Higher values are more accurate but slower. Default is 10. 
        - n_neighbors (int): The number of neighbors to search for in the Annoy index. Higher values are more accurate but slower. Default is 50.'''
        super().__init__()
        self.threshold = threshold
        self.debug_html = debug_html
        self.n_trees = n_trees
        self.n_neighbors = n_neighbors

    def eval(self, dataset: Dataset) -> Dataset:
        for image in dataset.images:
            if not image.clip.has_value:
                raise Abort("FuzzyDedupe requires images to have clip embeddings. Run ClipEmbedding first.")
        self.log.info("Building embedding matrix")
        dataset.images.sort(key=lambda image: image.objectid.value)
        embeddings = np.array([image.clip.value for image in dataset.images], dtype='float32')

        emb_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()

        params = {
            'threshold': self.threshold,
            'n_trees': self.n_trees,
            'n_neighbors': self.n_neighbors,
        }

        prop = CachedProperty('fuzzy_dedupe', 'v1', params, emb_hash)
        if prop.has_value:
            self.log.info("FuzzyDedupe has already been run with the same embedding matrix")
            old_images = len(dataset.images)
            dataset.images = [image for image in dataset.images if image.objectid.value in prop.value]
            self.log.info(f"Kept {len(dataset.images)} images from {old_images}")
            return dataset

        d = embeddings.shape[1]  # Dimensionality of embeddings

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / norms

        # Build an Annoy index for cosine similarity
        self.log.info("Adding items to Annoy index")
        annoy_index = AnnoyIndex(d, 'angular')  # 'angular' for cosine similarity

        for i, embedding in tqdm(enumerate(embeddings_normalized), total=len(embeddings_normalized), desc="Adding to Annoy"):
            annoy_index.add_item(i, embedding)

        # Build the index with n_trees (higher values are more accurate but slower)
        self.log.info("Building Annoy index")
        annoy_index.build(self.n_trees)

        self.log.info("Searching for similar images and clustering")
        uf = UnionFind(len(dataset.images))

        for i in tqdm(range(len(dataset.images)), desc="Clustering"):
            neighbors, distances = annoy_index.get_nns_by_item(i, self.n_neighbors, include_distances=True)
            for neighbor_idx, distance in zip(neighbors, distances):
                cosine_similarity = 1 - distance**2 / 2  # Convert angular distance to cosine similarity
                if cosine_similarity > self.threshold:
                    uf.union(i, neighbor_idx)

        # Aggregate clusters
        clusters_dict = {}
        for idx in range(len(dataset.images)):
            root = uf.find(idx)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(dataset.images[idx])

        # Create final clusters
        clusters = [cluster for cluster in clusters_dict.values()]

        self.log.info(f"Found {len(clusters)} clusters from {len(dataset.images)} images")
        if clusters:
            largest_cluster_size = max(len(cluster) for cluster in clusters)
            average_cluster_size = np.mean([len(cluster) for cluster in clusters])
            self.log.info(f"Largest cluster had {largest_cluster_size} images")
            self.log.info(f"Average cluster has {average_cluster_size:.2f} images")
        else:
            self.log.info("No clusters found.")

        multi_clusters  = [cluster for cluster in clusters if len(cluster) > 1]
        single_clusters = [cluster for cluster in clusters if len(cluster) == 1]

        if self.debug_html:
            # Create HTML file for multi-element clusters
            html_content = "<html><head><title>Fuzzy Dedupe Clusters</title></head><body>"
            html_content += "<h1>Multi-element Clusters</h1>"
            multi_clusters.sort(key=len, reverse=True)
            for idx, cluster in enumerate(multi_clusters, 1):
                html_content += f"<h2>Cluster {idx} ({len(cluster)} images)</h2><ul>"
                for image in cluster:
                    img_path = self.workspace.get_path(image)[len(self.workspace.dir):]
                    width = image.width.value
                    height = image.height.value
                    format = image.format.value
                    html_content += f'<li><img src="{img_path}" loading="lazy" alt="Image {image.objectid.value}" style="max-width:400px; max-height:400px; object-fit:contain;">'
                    html_content += f"<br>{width}x{height} {format}</li>"
                html_content += "</ul>"
            html_content += "</body></html>"

            html_path = Path(self.workspace.dir) / self.debug_html
            with open(html_path, 'w') as f:
                f.write(html_content)
            self.log.info(f"Clustering results saved to {html_path}")

        # Select single clusters and one image from each multi-cluster with largest area
        selected_images = [cluster[0] for cluster in single_clusters]
        for cluster in multi_clusters:
            largest_image = max(cluster, key=lambda img: img.width.value * img.height.value)
            selected_images.append(largest_image)

        # Cache the results for future runs
        prop.value = { image.objectid.value for image in selected_images }
        dataset.images = selected_images

        return dataset

__all__ = ['ExactDedupe', 'FuzzyDedupe']
