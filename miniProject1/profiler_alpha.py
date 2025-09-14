import pandas as pd

input_file = "miniProject1/Group4_Part1_preprocessed_roomtype 4 + amen.csv"
output_file = "miniProject1/Group1_Part1_Profile11.csv"

user_items = {
    1: [21, 70, 160, 440],
    2: [444, 579, 770],
    3: [751, 771, 779],
    4: [45, 84, 155],
    5: [81, 124, 702],
}

df = pd.read_csv(input_file)

results = []
for user_id, items in user_items.items():
    roomtypes = set()
    amenities = set()

    for item in items:
        matches = df[df["itemid"] == item]
        for _, row in matches.iterrows():
            roomtypes.add(str(row["roomtype_tokens_str"]).strip())
            amenities.update(
                a.strip(" '")
                for a in str(row["amenity_vector"]).strip("{}").split(",")
                if a.strip()
            )

    results.append(
        {
            "userID": user_id,
            "roomtypes": ", ".join(sorted(roomtypes)),
            "amenities": ", ".join(sorted(amenities)),
        }
    )

pd.DataFrame(results).to_csv(output_file, index=False)
print(f"Output saved to {output_file}")
