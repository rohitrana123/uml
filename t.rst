.. code:: ipython3

    # TSNE

.. code:: ipython3

    # Step 1: Import required Libraries
    # Students must complete the imports
    
    # TODO:---------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits

.. code:: ipython3

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


.. parsed-literal::

    (1797, 64)
    (1797,)
    original data shape: (1797, 64)
    


.. image:: output_2_1.png


.. code:: ipython3

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




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>...</th>
          <th>54</th>
          <th>55</th>
          <th>56</th>
          <th>57</th>
          <th>58</th>
          <th>59</th>
          <th>60</th>
          <th>61</th>
          <th>62</th>
          <th>63</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>5.0</td>
          <td>13.0</td>
          <td>9.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>6.0</td>
          <td>13.0</td>
          <td>10.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>12.0</td>
          <td>13.0</td>
          <td>5.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>11.0</td>
          <td>16.0</td>
          <td>10.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>15.0</td>
          <td>12.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>5.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3.0</td>
          <td>11.0</td>
          <td>16.0</td>
          <td>9.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>7.0</td>
          <td>15.0</td>
          <td>13.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>8.0</td>
          <td>...</td>
          <td>9.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>7.0</td>
          <td>13.0</td>
          <td>13.0</td>
          <td>9.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>11.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2.0</td>
          <td>16.0</td>
          <td>4.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 64 columns</p>
    </div>



.. code:: ipython3

    #Display summary statistics
    df.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>...</th>
          <th>54</th>
          <th>55</th>
          <th>56</th>
          <th>57</th>
          <th>58</th>
          <th>59</th>
          <th>60</th>
          <th>61</th>
          <th>62</th>
          <th>63</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>1797.0</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>...</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
          <td>1797.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>0.0</td>
          <td>0.303840</td>
          <td>5.204786</td>
          <td>11.835838</td>
          <td>11.848080</td>
          <td>5.781859</td>
          <td>1.362270</td>
          <td>0.129661</td>
          <td>0.005565</td>
          <td>1.993879</td>
          <td>...</td>
          <td>3.725097</td>
          <td>0.206455</td>
          <td>0.000556</td>
          <td>0.279354</td>
          <td>5.557596</td>
          <td>12.089037</td>
          <td>11.809126</td>
          <td>6.764051</td>
          <td>2.067891</td>
          <td>0.364496</td>
        </tr>
        <tr>
          <th>std</th>
          <td>0.0</td>
          <td>0.907192</td>
          <td>4.754826</td>
          <td>4.248842</td>
          <td>4.287388</td>
          <td>5.666418</td>
          <td>3.325775</td>
          <td>1.037383</td>
          <td>0.094222</td>
          <td>3.196160</td>
          <td>...</td>
          <td>4.919406</td>
          <td>0.984401</td>
          <td>0.023590</td>
          <td>0.934302</td>
          <td>5.103019</td>
          <td>4.374694</td>
          <td>4.933947</td>
          <td>5.900623</td>
          <td>4.090548</td>
          <td>1.860122</td>
        </tr>
        <tr>
          <th>min</th>
          <td>0.0</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>...</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>0.0</td>
          <td>0.000000</td>
          <td>1.000000</td>
          <td>10.000000</td>
          <td>10.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>...</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>1.000000</td>
          <td>11.000000</td>
          <td>10.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>0.0</td>
          <td>0.000000</td>
          <td>4.000000</td>
          <td>13.000000</td>
          <td>13.000000</td>
          <td>4.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>...</td>
          <td>1.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>4.000000</td>
          <td>13.000000</td>
          <td>14.000000</td>
          <td>6.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>0.0</td>
          <td>0.000000</td>
          <td>9.000000</td>
          <td>15.000000</td>
          <td>15.000000</td>
          <td>11.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>3.000000</td>
          <td>...</td>
          <td>7.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>10.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>12.000000</td>
          <td>2.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>max</th>
          <td>0.0</td>
          <td>8.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>15.000000</td>
          <td>2.000000</td>
          <td>16.000000</td>
          <td>...</td>
          <td>16.000000</td>
          <td>13.000000</td>
          <td>1.000000</td>
          <td>9.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
        </tr>
      </tbody>
    </table>
    <p>8 rows × 64 columns</p>
    </div>



.. code:: ipython3

    #Check whether missing values exist
    df.isna().sum()




.. parsed-literal::

    0     0
    1     0
    2     0
    3     0
    4     0
         ..
    59    0
    60    0
    61    0
    62    0
    63    0
    Length: 64, dtype: int64



.. code:: ipython3

    #Standardize the feature matrix using StandardScaler
    scaler = StandardScaler()
    #Store the transformed data in X_scaled
    X_scaled = scaler.fit_transform(X)
    print(X_scaled.shape)
    


.. parsed-literal::

    (1797, 64)
    

.. code:: ipython3

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
    #------------------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=25,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    
    X_tsne = tsne.fit_transform(X_scaled)
    print(X_tsne.shape)


.. parsed-literal::

    (1797, 2)
    

.. code:: ipython3

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



.. image:: output_8_0.png


.. code:: ipython3

    #Comparision of plots
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



.. image:: output_9_0.png


.. code:: ipython3

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
    


.. parsed-literal::

    (1797, 2)
    

.. code:: ipython3

    plt.figure()
    scatter = plt.scatter(X_tsne_2[:, 0], X_tsne_2[:, 1], c=y)
    plt.colorbar(scatter)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("t-SNE Visualization")
    plt.grid()
    plt.show()



.. image:: output_11_0.png


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
