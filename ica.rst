.. code:: ipython3

    # ICA

.. code:: ipython3

    # Step 1: Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import FastICA
    
    # Step 2: Generate source signals
    np. random.seed(42)
    
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    
    # Source signals
    s1 = np.sin(2 * time)
    s2 = np.sign(np.sin(3 * time))
    s3 = np.random.normal(size=n_samples)
    
    # Combine signals into matrix
    S = np.c_[s1, s2, s3]
    
    # Standardize signals
    S = S / S.std(axis=0)
    
    print("Source Shape:", S.shape)
    
    # Step 3: Plot original signals
    plt.figure(figsize=(10, 6))
    plt.plot(S[:, 0], label='Sine Signal' )
    plt.plot(S[:, 1], label='Square Signal' )
    plt.title("Original Source Signals")
    plt. legend ()
    plt.grid()
    plt.show()
    
    # Step 4: Mixing matrix
    A = np.array([[1, 1, 1],
    [0.5, 2, 1],
    [1.5, 1, 2]])
    
    # Create mixed signals
    X = np.dot(S, A.T)
    
    print("Mixed Shape:", X.shape)
    
    # Step 5: Plot mixed signals
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 0], label='Mixed 1')
    plt.plot(X[:, 1], label='Mixed 2' )
    plt.title("Mixed Signals")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Step 6: Apply ICA
    ica = FastICA(n_components=3, random_state=42)
    
    S_est = ica. fit_transform(X)
    A_est = ica.mixing_
    
    print("Recovered Shape:", S_est.shape)
    print("Estimated Mixing Matrix:\n", A_est)
    
    # Step 7: Plot recovered signals
    plt.figure(figsize=(10, 6))
    plt.plot(S_est[:, 0], label='Recovered 1')
    plt.plot(S_est[:, 1], label='Recovered 2' )
    plt.title("Recovered Signals using ICA")
    plt.legend ()
    plt.grid()
    plt.show()
    
    


.. parsed-literal::

    Source Shape: (2000, 3)
    


.. image:: output_1_1.png


.. parsed-literal::

    Mixed Shape: (2000, 3)
    


.. image:: output_1_3.png


.. parsed-literal::

    Recovered Shape: (2000, 3)
    Estimated Mixing Matrix:
     [[1.07951646 1.02801191 0.97309856]
     [0.60974459 1.99911967 0.98476982]
     [1.630728   1.02798866 1.9590621 ]]
    


.. image:: output_1_5.png


