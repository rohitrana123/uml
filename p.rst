.. code:: ipython3

    # PCA

.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 1: Import required libraries

    #TODO:
    #------------------------------------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_wine, load_breast_cancer, load_iris


.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 2: Load the dataset for this batch
    
    # TODO:
    # Keep one option active.
    
    # OPTION A: Wine dataset
    # data = load_wine()
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = pd.Series(data.target, name="target")
    
    # OPTION B: Breast Cancer dataset
    # data = load_breast_cancer()
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = pd.Series(data.target, name="target")
    
    # OPTION C: Custom CSV
    # df = pd.read_csv("batch_b_pca_dataset.csv")
    # X = df.drop(columns=["target"])
    # y = df["target"]
    
    # print("Feature shape:", X.shape)
    # X.head()
    #-------------------------------------------------------------------------
    iris = load_iris()
    
    x=iris.data
    y=iris.target
    
    feature_names = iris. feature_names
    target_name = iris.target_names
    
    print("original data shape:", x.shape)


.. code:: ipython3
   
    #------------------------------------------------------------------------
    #Optional - Visualizing the data (Extra)
    #------------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.scatter(x[:, 0], x[:, 1])
    plt.title('raw data (before pca)')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.grid(alpha=0.3)
    plt.show()


.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 3: Standardize the data
    
    # TODO:
    # scaler = ...
    # X_scaled = ...
    #--------------------------------------------------------------------
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    df = pd.DataFrame(x_scaled, columns = feature_names)
    df.head(10)


.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 4: Apply PCA using all components
    
    # TODO:
    # pca_full = ...
    # X_pca_full =
    # explained_var
    # cumulative_var = ...
    # print(explained_var)
    # print(cumulative_var)
    #------------------------------------------------------------------------
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(x_scaled)
    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print("Explained Variance Ratio per component:")
    print(explained_var)
    print("\nCumulative Explained Variance:")
    print(cumulative_var)
  

.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 5: Plot cumulative explained variance
    
    # TODO:
    # Create a line plot of cumulative explained variance
    # Add Labels, title, marker, and grid
    #------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance by PCA Components' )
    plt.xlabel('Number of Principal Components' )
    plt.ylabel('Cumulative Explained Variance' )
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(range(1, len(cumulative_var) +1))
    plt.show()


.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 6: Reduce data to two principal components
    
    # TODO:
    # pca_2
    # X_pca_2 = ..
    # print(X_pca_2.shape)
    # print("Variance captured:", ... )
    #---------------------------------------------------------------------
    
    pca_2 = PCA(n_components=2)
    X_pca_2 = pca_2.fit_transform(x_scaled)
    variance_captured = np.sum(pca_2.explained_variance_ratio_)
    
    print("Shape of reduced data:", X_pca_2.shape)
    print(f"Variance captured by 2 components: {variance_captured:.2%}")


.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 7: Plot the PCA-transformed data
    
    # TODO:
    # Create a scatter plot of PC1 vs PC2
    # Use target labels as colors
    # Add axis labels and title
    #--------------------------------------------------------------------------
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y, cmap='viridis', edgecolors='k')
    
    plt.title('PCA of Iris Dataset (2 Components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    handles, _ = scatter.legend_elements ()
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    plt.legend(handles, target_names, title="Species")
    plt.grid(alpha=0.3)
    plt.show()


.. code:: ipython3
    
    #------------------------------------------------------------------------
    # Step 8: Optional feature contribution analysis
    
    # TODO:
    # Create a DataFrame using pca_2.components_.T
    # Use original feature names as index
    # Identify the features contributing strongly to PC1 and PC2
    # Print the top positive / negative contributors
    #--------------------------------------------------------------------
    
    loadings = pd.DataFrame(
        pca_2.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    print("PCA Component Loadings:")
    print(loadings)
    
    print("\nTop contributors for PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False))
    
    print("\nTop contributors for PC2:")
    print(loadings['PC2'].abs().sort_values(ascending=False))


.. code:: ipython3

    ## Student Observation Section
    
    #   1)How much total variance is retained by the first two principal components?
    #   Ans: The first two principal components retain ~95.8% of total variance 
    #        (PC1: 72.9%, PC2: 22.9%), indicating minimal information loss.
    
    #   2)Which features contribute strongly to PC1 and PC2?
    #   Ans:PC1 is influenced by petal length, petal width, and sepal length (overall size).
    #       PC2 is mainly influenced by sepal width.
    
    #   3)Is class separation visible after projection?
    #   Ans: Yes, classes are clearly separated, especially Setosa.
    #        PC1 contributes most to the separation.
    
    #   4)Why is PCA usually applied after standardization?
    #   Ans: Ensures equal feature contribution
    #        Reduces dimensions
    #        Helps visualization
    #        Removes noise and correlation

