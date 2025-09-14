import pandas as pd
import math

visited_items = {
    1: [21, 70, 160, 440],
    2: [444, 579, 770],
    3: [751, 771, 779],
    4: [45, 84, 155],
    5: [81, 124, 702]
}

def cosine_similarity(vec1, vec2):
    # Common vocabulary unioned in both sets
    common_vocab = list(vec1.union(vec2))
    # Abuse space complexity
    vec1_binary = [1 if item in vec1 else 0 for item in common_vocab]
    vec2_binary = [1 if item in vec2 else 0 for item in common_vocab]
    # Calculate dot product
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1_binary, vec2_binary))
    # Calculate magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1_binary))
    mag2 = math.sqrt(sum(v * v for v in vec2_binary))
    if mag1 * mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def parse_set_string(set_str):
    # Parse a string representation of a set
    if pd.isna(set_str) or not isinstance(set_str, str):
        return set()
    # Remove curly braces and split by commas
    cleaned = set_str.strip().lower()
    if cleaned.startswith('{'):
        cleaned = cleaned[1:]
    if cleaned.endswith('}'):
        cleaned = cleaned[:-1]
    # Split and clean items
    items = [item.strip().strip("'\"") for item in cleaned.split(',')]
    return set(items)

# Load user preferences
user_df = pd.read_csv('user_roomtypes_amenities.csv')
# Load hotel data
hotels_df = pd.read_csv('Group4_Part1_preprocessed_roomtype 4 + amen.csv')

# Create mapping from itemid to all hotelids (to expand visited items)
item_to_hotels = {}
for _, row in hotels_df.iterrows():
    item_id = row['itemid']
    hotel_id = row['hotelid']
    if item_id not in item_to_hotels:
        item_to_hotels[item_id] = set()
    item_to_hotels[item_id].add(hotel_id)

# Expand visited items to visited hotel IDs for each user
visited_hotels_by_user = {}
for user_id, item_list in visited_items.items():
    visited_hotels_by_user[user_id] = set()
    for item_id in item_list:
        if item_id in item_to_hotels:
            visited_hotels_by_user[user_id].update(item_to_hotels[item_id])

# Parse amenity strings into sets
user_df['amenity_set'] = user_df['amenities'].apply(
    lambda x: set(item.strip().lower() for item in str(x).split(',')) if pd.notna(x) else set()
)
hotels_df['amenity_set'] = hotels_df['amenity_vector'].apply(parse_set_string)

# Create output dataframe
output_data = []

# Calculate cosine similarity for each hotel with each user
for hotel_idx, hotel_row in hotels_df.iterrows():
    item_id = hotel_row['itemid']
    hotel_id = hotel_row['hotelid']
    hotel_amenities = hotel_row['amenity_set']
    # Initialize row with item ID and hotel ID
    row_data = {'itemid': item_id, 'hotelid': hotel_id}
    # Calculate similarity with each user
    for user_idx, user_row in user_df.iterrows():
        user_id = user_idx + 1
        user_amenities = user_row['amenity_set']
        # Check if this hotel was visited by this user (any item from same hotel company)
        if hotel_id in visited_hotels_by_user[user_id]:
            similarity = "VISITED"
        else:
            similarity = cosine_similarity(user_amenities, hotel_amenities)
        row_data[f'user_{user_id}'] = similarity
    output_data.append(row_data)
# Create output dataframe
output_df = pd.DataFrame(output_data)
# Reorder columns: itemid, hotelid, user_1, user_2, user_3, user_4, user_5
columns = ['itemid', 'hotelid'] + [f'user_{i}' for i in range(1, 6)]
output_df = output_df[columns]
# Save to CSV
output_df.to_csv('hotel_user_similarity_matrix.csv', index=False)
print("Similarity matrix saved to 'hotel_user_similarity_matrix.csv'")
