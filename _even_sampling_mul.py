import random
from collections import Counter
import pandas as pd
from dataset import create_data


def preprocess_value(value):
    """Return the first element if the value is a list, else the value itself."""
    if isinstance(value, list) and value:
        return value[0] 
    elif isinstance(value, dict) and value:
        if len(value['name']) == 'w':
            return 'MISSING'
        return value['name']
    return value if value is not None else 'MISSING'


def find_top_two_values(data, key):
    """Find the two most common values for a given key."""
    counter = Counter(preprocess_value(item.get(key)) for item in data)
    if 'MISSING' in counter:
        del counter['MISSING']
    top_two = counter.most_common(2)
    return top_two

def sample_rows_for_values(data, key, values, n=500):
    """Sample n rows for each of the specified values."""
    samples = []
    for value in values:
        filtered = [item for item in data if preprocess_value(item.get(key)) == value]
        sampled = random.sample(filtered, min(n, len(filtered)))
        samples.extend(sampled)
    return samples


if __name__ == "__main__":
    # Data
    pretraining_movie_path = './data/pretraining_data_movie.jsonl'
    pretraining_product_path = './data/pretraining_data_product.jsonl'

    movie_path = './data/Movie_top100'
    product_path = './data/Product_top100'

    movie = create_data(movie_path, path_is="test", sample_num=1000000, pretraining_path=pretraining_movie_path)
    product = create_data(product_path, path_is="test", sample_num=1000000, pretraining_path=pretraining_product_path)

    # Step 1: Find top 2 genres for movie and top 2 categories for product
    top_two_genres = find_top_two_values(movie, 'genre')
    top_two_categories = find_top_two_values(product, 'category')

    print("Top two genres in movie:", top_two_genres)
    print("Top two categories in product:", top_two_categories)

    # Step 2: Extract the keys (genre/category names) for sampling
    top_movie_genres = [genre for genre, _ in top_two_genres]
    top_product_categories = [category for category, _ in top_two_categories]

    # Step 3: Sample 500 rows per genre for movies
    movie_samples = sample_rows_for_values(movie, 'genre', top_movie_genres, n=500)
    product_samples = sample_rows_for_values(product, 'category', top_product_categories, n=500)

    for i in range(len(movie_samples)):
        if isinstance(movie_samples[i]['genre'], list):
            movie_samples[i]['genre'] = movie_samples[i]['genre'][0]

    # Step 4: Convert sampled lists to DataFrames for easy analysis
    movie_df = pd.DataFrame(movie_samples)
    product_df = pd.DataFrame(product_samples)

    # Step 5: Save to CSV for further inspection
    movie_df.to_csv('movie_for_cls.csv', index=False)
    product_df.to_csv('product_for_cls.csv', index=False)

    # Step 6: Display results
    print(f"Sampled movie data shape: {movie_df.shape}")
    print(f"Sampled product data shape: {product_df.shape}")

    print("Movie genre distribution:\n", movie_df['genre'].value_counts())
    print("Product category distribution:\n", product_df['category'].value_counts())
