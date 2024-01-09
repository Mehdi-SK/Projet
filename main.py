import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.io.arff import loadarff
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
def preprocess_dataset(df):
    # standardize numeric data using StandardScaler
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # check for null columns and drop them
    null_cols = df.columns[df.isnull().all()]
    if len(null_cols) > 0:
        st.write(f"Dropping null columns: {list(null_cols)}")
        df.drop(columns=null_cols, inplace=True)

    # check for missing values within rows and propose a strategy to replace them
    missing_cols = df.columns[df.isnull().any()]
    for col in missing_cols:
        col_dtype = df[col].dtype
        if col_dtype == 'float' or col_dtype == 'int':
            st.write(f"Filling missing values in {col} with mean")
            df[col].fillna(df[col].mean(), inplace=True)
        elif col_dtype == 'object':
            st.write(f"Filling missing values in {col} with mode")
            df[col].fillna(df[col].mode()[0], inplace=True)

    # normalize categorical data using LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # return the preprocessed DataFrame
    return df


def find_optimal_k(data):
    distortions = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    # Use KneeLocator to find the optimal k
    knee = KneeLocator(list(K_range), distortions, curve='convex', direction='decreasing')

    # Create the plot using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(K_range, distortions, marker='o')
    ax.set(xlabel='Number of Clusters (k)', ylabel='Distortion',
           title='Elbow Method for Optimal k')
    ax.vlines(knee.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color='red', label='Optimal k')
    ax.legend()
    st.subheader("Elbow Curve")
    st.pyplot(fig)

    # Return the optimal k value
    optimal_k = knee.knee
    return optimal_k

def cluster(datframe, optimal_k, algorithm):
    df = datframe
    k = optimal_k
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    X = df[numeric_cols]
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=optimal_k, random_state=0, n_init='auto')
        labels = model.fit_predict(X)
        # fig = show_clusters(X, model)

        # plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig = plt.figure()
        # Plotting the clusters with PCA
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        plt.title(f'Clusters with PCA using Kmeans')
        st.pyplot(fig)
        
        # SIMILIARITY CALCULATIONS
        # Compute centroids
        centroids = model.cluster_centers_

        # Calculate inter-class similarity
        inter_class_sim = pairwise_distances(centroids)
        avg_inter_class_sim = inter_class_sim.mean()

        # Calculate intra-class similarity
        intra_class_sim = 0
        for cluster_id in range(optimal_k):
            cluster_points = X[labels == cluster_id]
            intra_class_sim += pairwise_distances(cluster_points).mean()
        avg_intra_class_sim = intra_class_sim / optimal_k
        st.write('Inter class similiarity = ', avg_inter_class_sim)
        st.write('Intra class similiarity = ', avg_intra_class_sim)
    elif algorithm == 'KMedoids':
        model = KMedoids(n_clusters=optimal_k, random_state=0)
        labels = model.fit_predict(X)
        # display
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Plotting the clusters with PCA
        fig = plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.title(f'Clusters with PCA Kmedoids')
        st.pyplot(plt)
        # distances
        distances = pairwise_distances(X, metric='euclidean')
        cluster_labels = model.labels_
        intra_similarity = 0
        
        for i in range(k):
            indices = np.where(cluster_labels == i)[0]
            if len(indices) > 1:
                cluster_dist = distances[np.ix_(indices, indices)]
                cluster_similarity = np.mean(cluster_dist)
                if not np.isnan(cluster_similarity):
                    intra_similarity += cluster_similarity
        intra_similarity /= k

        inter_similarity = 0
        for i in range(k):
            indices_i = np.where(cluster_labels == i)[0]
            for j in range(i + 1, k):
                indices_j = np.where(cluster_labels == j)[0]
                cluster_dist = distances[np.ix_(indices_i, indices_j)]
                cluster_similarity = np.mean(cluster_dist)
                if not np.isnan(cluster_similarity):
                    inter_similarity += cluster_similarity
        inter_similarity /= (k * (k - 1) / 2)


        st.write('Inter class similiarity = ', inter_similarity)
        st.write('Intra class similiarity = ', intra_similarity)

        
    elif algorithm == 'Agnes':
        distances = pairwise_distances(X, metric='euclidean')
        model = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete')
        labels = model.fit_predict(X)
        
        
        # display
        fig = plt.figure()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')

        plt.title('AGNES with PCA')
        st.pyplot(fig)

        
        # plot dendrogram
        linkage_matrix = linkage(X, method='complete')  # Calculate the linkage matrix
        fig = plt.figure()
        dendrogram(linkage_matrix, color_threshold=0, labels=labels)
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        st.pyplot(fig)

        # distances
        cluster_labels = model.labels_
        intra_dissimilarity = 0
        num_intra_pairs = 0
        for i in range(k):
            indices = np.where(cluster_labels == i)[0]
            if len(indices) > 1:
                cluster_dist = distances[np.ix_(indices, indices)]
                intra_dissimilarity += np.sum(cluster_dist)
                num_intra_pairs += len(indices) * (len(indices) - 1) // 2
        intra_dissimilarity /= num_intra_pairs

        # Step 6: Calculate inter-class dissimilarity
        inter_dissimilarity = 0
        num_inter_pairs = 0
        for i in range(k):
            indices_i = np.where(cluster_labels == i)[0]
            for j in range(i + 1, k):
                indices_j = np.where(cluster_labels == j)[0]
                cluster_dist = distances[np.ix_(indices_i, indices_j)]
                inter_dissimilarity += np.sum(cluster_dist)
                num_inter_pairs += len(indices_i) * len(indices_j)
        inter_dissimilarity /= num_inter_pairs
        
        st.write("Inter-class similarity:", intra_dissimilarity)
        st.write("Intra-class similarity:", inter_dissimilarity)
    elif algorithm == 'Diana':
        # model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        # labels = model.fit_predict(X)
        # fig = show_clusters(X, model)
        # st.pyplot(fig)
        # # plot dendrogram
        # Z = linkage(X, 'ward')
        # fig = plt.figure(figsize=(12, 6))
        # dn = dendrogram(Z)
        # plt.title('Dendrogram')
        # plt.xlabel('Samples')
        # plt.ylabel('Distance')
        # st.pyplot(fig)
        # Step 1: Compute pairwise distances
        distances = pairwise_distances(X, metric='euclidean')

        # Step 2: Initialize the clustering model with n_clusters=1
        model = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete')

        # Step 3: Perform divisive clustering
        labels = model.fit_predict(X)

        # Step 4: Display results with PCA
        fig = plt.figure()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,  cmap='viridis')
        plt.title('DIANA with PCA')
        st.pyplot(fig)

        # Step 5: Plot dendrogram
        linkage_matrix = linkage(X, method='single')  # Calculate the linkage matrix
        fig = plt.figure()
        dendrogram(linkage_matrix, color_threshold=0)
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        st.pyplot(fig)

        # Step 6: Calculate inter-class dissimilarity
        cluster_labels = model.labels_
        intra_dissimilarity = 0
        num_intra_pairs = 0
        k = len(np.unique(cluster_labels))
        for i in range(k):
            indices = np.where(cluster_labels == i)[0]
            if len(indices) > 1:
                cluster_dist = distances[np.ix_(indices, indices)]
                intra_dissimilarity += np.sum(cluster_dist)
                num_intra_pairs += len(indices) * (len(indices) - 1) // 2
        intra_dissimilarity /= num_intra_pairs

        # Step 7: Calculate intra-class dissimilarity
        inter_dissimilarity = 0
        num_inter_pairs = 0
        for i in range(k):
            indices_i = np.where(cluster_labels == i)[0]
            for j in range(i + 1, k):
                indices_j = np.where(cluster_labels == j)[0]
                cluster_dist = distances[np.ix_(indices_i, indices_j)]
                inter_dissimilarity += np.sum(cluster_dist)
                num_inter_pairs += len(indices_i) * len(indices_j)
        inter_dissimilarity /= num_inter_pairs
        st.write("Inter-class similarity:", intra_dissimilarity)
        st.write("Intra-class similarity:", inter_dissimilarity)
    elif algorithm == 'DBSCAN':
        # Step 1: Compute pairwise distances
        distances = pairwise_distances(X, metric='euclidean')

        # Step 2: Initialize and fit the DBSCAN model
        model = DBSCAN(eps=3, min_samples=2)
        labels = model.fit_predict(X)

        # Step 3: Display results with PCA
        fig = plt.figure()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.title('DBSCAN with PCA')
        st.pyplot(fig)

        # Step 4: Calculate the number of clusters (-1 indicates noise)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.write("Number of clusters:", num_clusters)

        # Step 5: Calculate inter-class dissimilarity
        cluster_labels = model.labels_
        intra_dissimilarity = 0
        num_intra_pairs = 0
        k = len(np.unique(cluster_labels))
        for i in range(k):
            indices = np.where(cluster_labels == i)[0]
            if len(indices) > 1:
                cluster_dist = distances[np.ix_(indices, indices)]
                intra_dissimilarity += np.sum(cluster_dist)
                num_intra_pairs += len(indices) * (len(indices) - 1) // 2

        # Check for division by zero
        if num_intra_pairs > 0:
            intra_dissimilarity /= num_intra_pairs

        # Step 6: Calculate intra-class dissimilarity
        inter_dissimilarity = 0
        num_inter_pairs = 0
        for i in range(k):
            indices_i = np.where(cluster_labels == i)[0]
            for j in range(i + 1, k):
                indices_j = np.where(cluster_labels == j)[0]
                cluster_dist = distances[np.ix_(indices_i, indices_j)]
                inter_dissimilarity += np.sum(cluster_dist)
                num_inter_pairs += len(indices_i) * len(indices_j)

        # Check for division by zero
        if num_inter_pairs > 0:
            inter_dissimilarity /= num_inter_pairs

        # Step 7: Display inter-class and intra-class dissimilarities
        st.write("Inter-class similarity:", intra_dissimilarity)
        st.write("Intra-class similarity:", inter_dissimilarity)
    else:
        raise ValueError(f'Invalid algorithm: {algorithm}')

    # display clustered data
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    st.write("Clustered Data:")
    st.write(df_clustered)

    

    return optimal_k



def classify(dataframe, algorithm):
    # Separate features and target variable
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier
    if algorithm == 'KNN':
        if len(set(y)) > 2:
            st.error("KNN works only for binary classification. Please choose a binary classification dataset.")
            return
        n_neighbors = st.sidebar.slider("Select the number of neighbors for KNN", 1, 20, 5)
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif algorithm == 'Naive Bayes':
        classifier = GaussianNB()
    elif algorithm == 'Decision Tree':
        # Feature selection
        criterion = st.sidebar.selectbox("Select criterion", ["gini", "entropy"])
        splitter = st.sidebar.selectbox("Select splitter", ["best", "random"])
        max_depth = st.sidebar.slider("Select max depth", 1, 20, 10)
        min_samples_split = st.sidebar.slider("Select min samples split", 2, 10, 2)
        min_samples_leaf = st.sidebar.slider("Select min samples leaf", 1, 10, 1)
        max_leaf_nodes = st.sidebar.slider("Select max leaf nodes", 2, 100, 50)

        classifier = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
        )
    elif algorithm == 'Neural Networks':
        # Feature selection
        hidden_layer_sizes = st.sidebar.text_input("Enter the sizes of hidden layers separated by commas (e.g., 100,50)", "100,50")
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
        max_iter = st.sidebar.slider("Select the maximum number of iterations for training", 100, 1000, 200)
        classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    elif algorithm == 'SVM':
        classifier = SVC(probability=True)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")

    # Display ROC curve and AUC score for binary classification
    # Display ROC curve
    fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    st.subheader("ROC Curve")
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    # Display the resulting tree for Decision Tree
    if algorithm == 'Decision Tree':
        st.subheader("Decision Tree Visualization")
        plt.figure(figsize=(12, 8))  # Adjust the figure size if needed
        plot_tree(classifier, filled=True, feature_names=X.columns, class_names=[str(c) for c in classifier.classes_], rounded=True)
        st.pyplot(plt.gcf())  # Show the figure using Streamlit





if __name__ == '__main__':
    # Dataset names
    datasets = {
        "diabetes": "./datasets/diabetes.arff",
        "hepatitis": "./datasets/hepatitis.arff",
        "ecoli": "./datasets/ecoli2.arff",
    }
    # Menu selections
    dataset_name = st.sidebar.selectbox("Select a dataset", list(datasets.keys()))
    raw_data = loadarff(datasets[dataset_name])
    df = pd.DataFrame(raw_data[0])
    df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Display datset
    st.subheader("Original dataset")
    st.write(df)

    preprocessed_df = preprocess_dataset(df)
    # display the preprocessed DataFrame
    st.subheader("After Preprocessing")
    st.write(preprocessed_df)

    # Algorithmes
    algorithm_type = st.sidebar.radio("Select algorithm type", ['Unsupervised', 'Supervised'])

    if algorithm_type == 'Unsupervised':
        algorithms = ['K-Means', 'KMedoids', 'Agnes', 'Diana', 'DBSCAN']
        algorithm = st.sidebar.selectbox("Select a clustering algorithm", algorithms)

        features_for_clustering = preprocessed_df.iloc[:, :-1]
        optimal_k = find_optimal_k(features_for_clustering)
        st.write(f"Optimal K for {algorithm}: {optimal_k}")
        cluster(datframe=features_for_clustering, optimal_k=optimal_k, algorithm=algorithm)

    elif algorithm_type == 'Supervised':
        algorithms = ['KNN', 'Naive Bayes', 'Decision Tree', 'Neural Networks', 'SVM']
        algorithm = st.sidebar.selectbox("Select a clustering algorithm", algorithms)

        classify(dataframe=preprocessed_df, algorithm=algorithm)
    else:
        st.error("Invalid algorithm type selected.")

    