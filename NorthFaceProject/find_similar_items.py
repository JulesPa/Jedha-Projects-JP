import pandas as pd
import random


def find_similar_items(data, item_id):

    cluster_label = data.loc[data['id'] == item_id, 'cluster'].values
    if len(cluster_label) == 0:
        print("Item ID not found.")
        return []
    
    cluster_label = cluster_label[0]
    
    similar_items = data[data['cluster'] == cluster_label]['id']
    similar_items = similar_items[similar_items != item_id]
    
    similar_items = random.sample(similar_items.tolist(), min(5, len(similar_items)))
    
    return similar_items


def main():
    
    data = pd.read_csv('processed_data.csv')  
    
    try:
        item_id = int(input("Enter the product ID you want to find similar items for: "))
        suggestions = find_similar_items(data, item_id)
        
        if suggestions:
            print(f"Similar items to product ID {item_id}: {suggestions}")
        else:
            print("No similar items found or item ID may be unique in its cluster.")
    except ValueError:
        print("Invalid input. Please enter a valid product ID.")

if __name__ == "__main__":
    main()