import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json  # Ensure json is imported


# --- K-Means Function ---
def run_kmeans_on_iris():
    """
    Loads Iris dataset, runs K-Means clustering, and returns processed data.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Using Sepal Length and Sepal Width for simplicity and visualization
    X_features = X[:, [0, 1]]

    # Ensure n_init is explicitly set to avoid future warnings/errors
    # n_init='auto' can be used for newer versions, but a number like 10 is safer for broader compatibility.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_features)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Create a DataFrame for easier handling
    df = pd.DataFrame({
        'sepal_length': X_features[:, 0],
        'sepal_width': X_features[:, 1],
        'cluster': labels,
        'species': y  # Actual species labels
    })

    # Convert species numbers to names for better readability in plotting
    species_names = [iris.target_names[i] for i in y]
    df['species_name'] = species_names

    # Convert centers to a list of lists for JSON serialization
    centers_list = centers.tolist()

    return df.to_dict(orient='records'), centers_list


# --- FastAPI Setup ---
app = FastAPI()

# For serving HTML and JS
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page with the plotting logic.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/iris_clusters")
async def get_iris_clusters():
    """
    API endpoint to get Iris clustering data.
    """
    data_records, centers_data = run_kmeans_on_iris()

    # Prepare the response payload
    response_payload = {
        "cluster_data": data_records,
        "cluster_centers": centers_data
    }

    return response_payload

