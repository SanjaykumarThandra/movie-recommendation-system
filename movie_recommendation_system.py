import pandas as pd
from surprise import Reader, Dataset
from surprise import KNNWithCosine, accuracy
from surprise.model_selection import train_test_split

# Step 1: Load the dataset
file_path = r"C:\Users\DELL\Downloads\movielens_data.csv"
data = pd.read_csv(file_path)

# Show the first few rows of the dataset to understand its structure
print("Dataset preview:")
print(data.head())

# Step 2: Prepare the dataset for Surprise
# Assuming the dataset contains columns: 'userId', 'movieId', 'rating'
reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Step 3: Split the data into training and test sets
trainset, testset = train_test_split(dataset, test_size=0.25)

# Step 4: Use k-NN algorithm with Cosine similarity
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based collaborative filtering
}

model = KNNWithCosine(sim_options=sim_options)
model.fit(trainset)

# Step 5: Make predictions on the test set
predictions = model.test(testset)

# Step 6: Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"RMSE (Root Mean Squared Error): {rmse}")

# Step 7: Example: Get recommendations for a specific user
user_id = 1  # Specify the user ID you want recommendations for
n_recommendations = 3  # Number of recommendations

# Get a list of all movies in the dataset
all_movies = data['movieId'].unique()

# Get the list of movies the user has already rated
user_rated_movies = data[data['userId'] == user_id]['movieId'].tolist()

# Filter out the movies the user has already rated
movies_to_predict = [movie for movie in all_movies if movie not in user_rated_movies]

# Predict ratings for the remaining movies
predictions_for_user = [model.predict(user_id, movie) for movie in movies_to_predict]

# Sort predictions by estimated rating and select top n recommendations
predictions_for_user.sort(key=lambda x: x.est, reverse=True)

# Step 8: Output the top n recommendations
print(f"Top {n_recommendations} recommendations for user {user_id}:")
for i in range(n_recommendations):
    print(f"Movie ID: {predictions_for_user[i].iid}, Predicted Rating: {predictions_for_user[i].est:.2f}")
