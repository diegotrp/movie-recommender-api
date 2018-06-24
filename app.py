from flask import Flask, request, Response
from flask_cors import CORS
from sklearn.externals import joblib
from collections import defaultdict
import pandas as pd
from surprise import Reader, Dataset, SVD, evaluate, accuracy

app = Flask(__name__)
CORS(app)

@app.route("/movies", methods=['GET'])
def top_movies():
	mdf = joblib.load("dumps/movies.pkl")
	return Response(mdf.to_json(orient='records'), mimetype='application/json')

@app.route("/movie_detail", methods=['GET'])
def movie_detail():
	mdf = joblib.load("dumps/movies.pkl")
	movie_id = request.args.get('movie_id')
	mdf = mdf[mdf['id'] == int(movie_id)]
	return Response(mdf.to_json(orient='records'), mimetype='application/json')

@app.route("/top_movies", methods=['GET'])
def movies():
	mdf = joblib.load("dumps/movies.pkl").head(10)
	return Response(mdf.to_json(orient='records'), mimetype='application/json')

@app.route("/recommendations", methods=['GET'])
def recommendations():
	movie_id = request.args.get('movie_id')
	cosine_sim = joblib.load("dumps/cosine_sim.pkl")
	movies = joblib.load("dumps/movies.pkl")
	movies['id'] = movies['id'].astype('int')

	indices = pd.Series(movies.index, index=movies['id'])
	idx = indices[int(movie_id)]

	sim_scores = list(enumerate(cosine_sim[int(idx)]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:31]
	movie_indices = [i[0] for i in sim_scores]
	scr = movies.iloc[movie_indices]
	return Response(scr.to_json(orient='records'), mimetype='application/json')
	
@app.route("/collaborative", methods=['POST'])
def collaborative():
	data = request.get_json()
	new_movies = data['movies']
	print(new_movies)

	movies = joblib.load("dumps/movies.pkl")
	movies['id'] = movies['id'].astype('int')
	ratings = joblib.load("dumps/ratings.pkl")
	ratings['rating'] = ratings['rating'].astype('int')

	reader = Reader(rating_scale=(1, 5))

	new_ratings = []
	for iid in new_movies:
	    new_ratings.append([9999, iid, 5])
	newUser = pd.DataFrame(new_ratings, columns=['userId', 'movieId', 'rating'])
	ratings = ratings.append(newUser, ignore_index=True)

	data = Dataset.load_from_df(ratings, reader)
	svd = SVD()
	trainset = data.build_full_trainset()
	svd.fit(trainset)

	testset = trainset.build_anti_testset()
	testset = list(filter(lambda x: x[0] == 9999, testset))
	predictions = svd.test(testset)
	top_n = []
	for uid, iid, true_r, est, _ in predictions:
	    top_n.append((iid, est))

	top_n = sorted(top_n, key=lambda x: x[1], reverse=True)
	top_n = top_n[1:10]
	fmovies = [i for i,j in top_n]

	df = pd.DataFrame()
	for iid in fmovies:
	    tmovie = movies[movies['id'] == iid]
	    ntp = pd.DataFrame(tmovie, columns=['id','title','description','genres','director','cast','url'])
	    df = pd.concat([df,ntp])

	return Response(df.to_json(orient='records'), mimetype='application/json')

if __name__ == '__main__':
	app.run(debug=True)