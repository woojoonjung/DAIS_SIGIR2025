import fasttext
import json

# Load FastText language model (download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html)
model_path = './lid.176.bin'
model = fasttext.load_model(model_path)

def fasttext_detect_language(text):
    """Detect the language of a given text using FastText."""
    if not isinstance(text, str):
        text = str(text)
    try:
        lang, _ = model.predict(text)
        return lang[0].replace('__label__', '')
    except Exception:
        return 'unknown'

def filter_english_rows(data):
    """Filter out rows that are not in English."""
    filtered_data = []

    for item in data:
        language = fasttext_detect_language(item)
        if language == 'en':
            filtered_data.append(item)

    return filtered_data

def save_to_jsonl(data, file_path):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Filtered data saved to {file_path}")

if __name__ == "__main__":
    # Data
    pretraining_movie_path = './data/pretraining_data_movie.jsonl'
    pretraining_product_path = './data/pretraining_data_product.jsonl'

    movie_path = './data/Movie_top100'
    product_path = './data/Product_top100'

    # Load data
    movie = create_data(movie_path, path_is="test", sample_num=1000000, pretraining_path=pretraining_movie_path)
    product = create_data(product_path, path_is="test", sample_num=1000000, pretraining_path=pretraining_product_path)

    # Filter datasets to keep only English rows
    movie_filtered = filter_english_rows(movie)
    product_filtered = filter_english_rows(product)

    # Save filtered datasets to JSONL files
    save_to_jsonl(movie_filtered, './data/movie_english.jsonl')
    save_to_jsonl(product_filtered, './data/product_english.jsonl')