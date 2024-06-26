# Utilize FAISS to Search Engine Optimization

This script processes a JSON file of product data, transforms and analyzes the data using pandas, generates embeddings for text data using the SentenceTransformers library, and performs search queries on the embeddings using FAISS.

FAISS is essential for a food product app because it offers fast, accurate, and scalable similarity search capabilities. Its ability to handle large datasets and integrate with machine learning models makes it a powerful tool for enhancing search queries, ultimately improving the user experience and satisfaction.

## Prerequisites
- Python 3.x
- `pandas` library
- `sentence-transformers` library
- `faiss` library

## Steps

1. **Load and Parse JSON Data**:
   - Reads the JSON file `Ecocheck_Data.json`.
   - Handles JSONDecodeError and reads line by line if necessary.
   ```python
   data = load_json_file('Ecocheck_Data.json')
   df = pd.DataFrame(data)
   true_df = df[df['labels'].apply(lambda x: len(x) > 0)]
   ```

2. **Convert Categorical Data**:
   - Processes the `organic_label` and `vegan_label` columns.
   - Converts lists to strings and sets custom sort keys.
   ```python
   true_df['organic_label'] = true_df['organic_label'].apply(process_labels)
   true_df['vegan_label'] = true_df['vegan_label'].apply(process_labels)
   true_df['organic_label'] = true_df['organic_label'].apply(lambda x: 1 if 'Bio' in x else 0)
   true_df['vegan_label'] = true_df['vegan_label'].apply(lambda x: 1 if 'Vegan' in x else 0)
   ```

3. **Convert Price Data**:
   - Cleans and converts the `price` column to numeric.
   ```python
   true_df['price'] = true_df['price'].str.replace('[^\d.]', '', regex=True)
   true_df['price'] = pd.to_numeric(true_df['price'], errors='coerce')
   ```

4. **Combine Columns**:
   - Combines multiple text columns into a single column for embedding.
   ```python
   true_df['combined'] = true_df.apply(combine_columns, axis=1)
   ```

5. **Generate Embeddings**:
   - Loads a multilingual model from SentenceTransformers.
   - Generates embeddings for the combined text column.
   ```python
   model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
   sentences = true_df['combined'].tolist()
   embeddings = model.encode(sentences)
   ```

6. **Indexing with FAISS**:
   - Creates and trains a FAISS index for the embeddings.
   ```python
   index = indexing(embeddings, 100)
   ```

7. **Search and Sort**:
   - Performs a search query on the FAISS index.
   - Sorts the search results based on specified columns.
   ```python
   most_relevant_columns = ['organic_label', 'vegan_label']
   search_query = "chicken"
   Best_results, running_time = SearchInputQuery(search_query, 50, index, 5, most_relevant_columns)
   ```

8. **Output Results**:
   - Prints the search query and execution time.
   - Displays the top search results.
   ```python
   print(f"Execution time: {running_time*1e3} milli-seconds")
   print(f"Search Query: {search_query} \n")
   Best_results[['name','labels']]
   ```
