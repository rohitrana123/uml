.. code:: ipython3

    # TSNE

.. code:: ipython3
    #------------------------------------------------------------------------
    # Step 1: Import required Libraries
    
    # TODO:---------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits
    
.. code:: ipython3
    #------------------------------------------------------------------------
    # Step 2: Load the dataset for this batch
    
    # TODO:
    # Keep only one option active.
    
    # #OPTION A: Digits dataset
    # data = Load_digits()
    # X = pd.DataFrame(data.data)
    # y = pd. Series(data.target, name="target")
    
    # OPTION B: Wine dataset
    # data = Load_wine()
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = pd.Series(data.target, name="target")
    
    # OPTION C: Breast Cancer dataset
    # data = Load_breast_cancer()
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = pd.Series(data.target, name="target")
    
    # OPTION D: Custom CSV
    # df = pd.read_csv("batch_b_tsne_dataset.csv")
    # X = df.drop(columns=["target"])
    # y = df["target"]
    
    # print(X.shape)
    # print(y.shape)
    # X.head()
    #------------------------------------------------------------------------
    #Using Digits Dataset
    data = load_digits()
    X = data.data
    y = data.target
    
    print(X.shape)
    print(y.shape)
    
    feature_names = data.feature_names
    target_names = data.target_names
    
    print("original data shape:", X.shape)
    # plot 
    plt.figure(figsize=(5, 4))
    
    plt.scatter(X[:, 0], X[:, 1])
    
    plt.title('raw data (before t-sne)')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    
    plt.grid(alpha=0.3)
    plt.show()


.. code:: ipython3
    #------------------------------------------------------------------------
    # Step 3: Inspect and preprocess the dataset
    
    # TODO:
    # 1. Display summary statistics
    # 2. Check whether missing values exist
    # 3. Standardize the feature matrix using StandardScaler
    # 4. Store the transformed data in X_scaled
    #-------------------------------------------------------------------
    #Convert it into Dataframe
    df = pd.DataFrame(X)
    df.head()


.. code:: ipython3
    
    #Display summary statistics
    df.describe()
    

.. code:: ipython3

    #Check whether missing values exist
    df.isna().sum()


.. code:: ipython3

    #Standardize the feature matrix using StandardScaler
    scaler = StandardScaler()
    #Store the transformed data in X_scaled
    X_scaled = scaler.fit_transform(X)
    print(X_scaled.shape)
    

.. code:: ipython3
    #------------------------------------------------------------------------
    # Step 4: Create and apply the t-SNE model
    
    # Suggested parameters for this batch:
    # n_components = 2
    # perplexity = 25
    # Learning_rate = 'auto'
    # init = 'pca'
    # random_state = 42
    
    # TODO:
    # tsne = .
    # X_tsne =
    # print(X_tsne.shape)
    #------------------------------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=25,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    
    X_tsne = tsne.fit_transform(X_scaled)
    print(X_tsne.shape)


.. code:: ipython3
    #------------------------------------------------------------------------
    # Step 5: Plot the transformed data
    
    # TODO:
    # 1. Create a scatter plot
    # 2. Use class labels for color-coding
    # 3. Add title, axis labels, and grid
    # 4. Show the plot
    #----------------------------------------------------------------------
    plt.figure()
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.colorbar(scatter)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("t-SNE Visualization")
    plt.grid()
    plt.show()


.. code:: ipython3
    #------------------------------------------------------------------------
    #Comparision of plots
    #------------------------------------------------------------------------
    plt.figure(figsize=(10,5))
    
    # Original data
    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    # t-SNE data
    plt.subplot(1,2,2)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)
    plt.title("t-SNE Data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    plt.tight_layout()
    plt.show()


.. code:: ipython3
    #------------------------------------------------------------------------
    # Optional Challenge
    # Repeat the experiment with another perplexity value
    # such as 10 or 40, then compare both embeddings.
    
    # TODO:
    # perplexity_2 = ...
    # tsne_2 =
    # X_tsne_2 = ...
    # plot and compare
    #-------------------------------------------------------------
    #Using perplexity as 40
    tsne_2 = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    
    X_tsne_2 = tsne_2.fit_transform(X_scaled)
    print(X_tsne_2.shape)
    

.. code:: ipython3
    #------------------------------------------------------------------------
    plt.figure()
    scatter = plt.scatter(X_tsne_2[:, 0], X_tsne_2[:, 1], c=y)
    plt.colorbar(scatter)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("t-SNE Visualization")
    plt.grid()
    plt.show()
    #------------------------------------------------------------------------


.. code:: ipython3

    #Stusent Observation
    #    1)Are the classes or groups clearly separated?
    #    Ans: We can see distinct cluster of the digits they are clearly seperated.
        
    #    2)Which points seem to overlap or form confusing regions?
    #    Ans: Few points overlap because it groups fews digits like 3 & 5 together because they have similar structure. T-SNE preserve the global
    #    structure of the data.
        
    #    3)How does t-SNE differ from PCA for visualization?
    #    Ans: t-SNE is non linear dimensionality reduction which is a visualization technique used to visualize high dimensional data into 2d or 3d.
    #   Pca captures maximum variance, tsne reduces the dimension
        
    #   4)How does changing perplexity influence the embedding?
    #    Ans: changing perplexity the cluster become more defined and clearly seperated from each other. but the value has slightly changed. the
    #    points are shifting.
