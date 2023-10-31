from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
import mysql.connector

def get_rating():
    con = mysql.connector.connect(
        user='root',
        password='',
        host='localhost',
        database='toko-online'
    )

    cursor = con.cursor()
    query = 'SELECT id, user_id, product_id, rating FROM rating'
    cursor.execute(query)

    result = []
    for i, data in enumerate(cursor):
        result.append(data)
    cursor.close()
    con.close()

    df = pd.DataFrame(result)
    df.columns = ['id', 'user_id', 'product_id', 'rating']
    return df

def get_product():
    con = mysql.connector.connect(
        user='root',
        password='',
        host='localhost',
        database='toko-online'
    )

    cursor = con.cursor()
    query = 'SELECT id, name, image FROM products'
    cursor.execute(query)

    result = []
    for i, data in enumerate(cursor):
        result.append(data)
    cursor.close()
    con.close()

    df = pd.DataFrame(result)
    df.columns = ['product_id', 'name', 'image']
    return df


def get_user():
    con = mysql.connector.connect(
        user='root',
        password='',
        host='localhost',
        database='toko-online'
    )

    cursor = con.cursor()
    query = 'SELECT id, email FROM users'
    cursor.execute(query)

    result = []
    for i, data in enumerate(cursor):
        result.append(data)
    cursor.close()
    con.close()

    df = pd.DataFrame(result)
    df.columns = ['user_id', 'email']
    return df


def compute_recomender(product, rating, user):
    df = pd.merge(rating, product, on='product_id', how='inner')
    df = pd.merge(df, user, on='user_id', how='inner')

    utility = df.pivot(index = 'name', columns = 'email', values = 'rating')
    utility = utility.fillna(0)
    
    a = rating.groupby('product_id')
    b = a.first()

    # similarity = 1- distance
    distance_mtx = squareform(pdist(utility, 'cosine'))
    similarity_mtx = 1- distance_mtx

    # item_similarity = utility.T.corr()
    # similarity_mtx = item_similarity.to_numpy()

    return similarity_mtx, utility, b

def calculate_user_rating(email, similarity_mtx, utility):
    user_rating = utility.loc[:,email]
    pred_rating = deepcopy(user_rating)
     
    default_rating = user_rating[user_rating>0].mean()
    numerate = np.dot(similarity_mtx, user_rating)
    corr_sim = similarity_mtx[:, user_rating >0]
    for i,ix in enumerate(pred_rating):
        temp = 0
        if ix < 1:
            w_r = numerate[i]
            sum_w = corr_sim[i,:].sum()
            if w_r == 0 or sum_w == 0:
                temp = default_rating
            else:
                temp = w_r / sum_w
            pred_rating.iloc[i] = temp
    return pred_rating


def recommendation_to_user(email, top_n, similarity_mtx, utility, b):
    user_rating = utility.loc[:,email]
    pred_rating = calculate_user_rating(email, similarity_mtx, utility)

    top_item = sorted(range(1,len(pred_rating)), key = lambda i: -1*pred_rating.iloc[i])
    top_item = list(filter(lambda x: user_rating.iloc[x]==0, top_item))[:top_n]
    res = []
    for i in top_item:
        res.append({"id": b['id'].index[i].item(), "pred" : pred_rating.iloc[i]})

    return res

app = Flask(__name__)
product = get_product()
product.rename(columns={'product_id': 'id'}, inplace=True)

@app.route('/predict', methods=['POST'])
def predict():
    email = request.json["email"]
    print(f"email adalah : {email}")
    # top_n = request.form["ton_n"]
    rating = get_rating()
    productsql = get_product()
    user = get_user()

    sm, ut, b = compute_recomender(productsql, rating, user)
    predict = recommendation_to_user(email, 30 , sm, ut, b)
    responses = []
    for i in predict:
        image = product.loc[product['id'] == i['id']]['image'].values[0]
        name = product.loc[product['id'] == i['id']]['name'].values[0]

        responses.append({'id': i['id'], "name": name, "image": image})

    return jsonify(responses)


if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='0.0.0.0', port=5000)