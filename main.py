import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json  # Ensure json is imported


# --- K-Means تابع ---
def run_kmeans_on_iris():
    """
    دیتاست ایریس لود می کنیم
    خوشه بندی را ران می کنیم و دیتایپردازش شده را برمی گرداند

    """
    iris = load_iris()
    X = iris.data
    y = iris.target


    X_features = X[:, [0, 1]]


    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_features)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_


    df = pd.DataFrame({
        'sepal_length': X_features[:, 0],
        'sepal_width': X_features[:, 1],
        'cluster': labels,
        'species': y
    })


    species_names = [iris.target_names[i] for i in y]
    df['species_name'] = species_names


    centers_list = centers.tolist()

    return df.to_dict(orient='records'), centers_list



app = FastAPI()


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
       را ایجاد می کنم HTML page
    """
    return templates.TemplateResponse("index.html", {"request": request})




@app.get("/api/iris_clusters")
async def get_iris_clusters():
    """
    اندپوینت ای پی ای برای گرفتن دیتای Iris

        """
    data_records, centers_data = run_kmeans_on_iris()

    
    response_payload = {
        "cluster_data": data_records,
        "cluster_centers": centers_data
    }

    return response_payload

