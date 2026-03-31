.. code:: ipython3

    # ICA

.. code:: ipython3

    # Step 1: Import required libraries
    
    # TODO:
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.decomposition import FastICA
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import FastICA
    # Step 2: Generate or define source signals
    
    # Suggested idea:
    # - sine signal
    # - square-like signal
    # - noise signal
    
    # TODO:
    # np.random.seed(...)
    # n_samples = ...
    # time = ...
    # s1 = ...
    # s2 = ...
    # s3 = ...
    # S = np.c_[s1, s2, s3]
    # S = S / S.std(axis=0)
    # print(S.shape)
    
    np.random.seed(42)
    n_samples = 1000
    time = np.linspace(0,8,n_samples)
    s1 = np.sin(2*time)
    s2 = np.sign(np.sin(3*time))
    s3 = np.random.normal(size=n_samples)
    S = np.c_[s1, s2, s3]
    S = S / S.std(axis=0)
    print(S.shape)
    
    # Step 3: Plot original source signals
    
    # TODO:
    # Create a line plot for each source signal
    # Add legend, title, labels, and grid
    
    print("Original Source Signals")
    fig,axes = plt.subplots(3,1,figsize=(10,6), sharex=True)
    
    axes[0].plot(time,S[:,0])
    axes[0].set_title("Sine Wave")
    
    axes[1].plot(time,S[:,1])
    axes[1].set_title("Square Wave")
    
    axes[2].plot(time,S[:,2])
    axes[2].set_title("Noise")
    
    plt.xlabel("Time")
    plt.show()
    
    # Step 4: Define a mixing matrix and create mixed signals
    
    # TODO:
    # A = np.array([...])
    # X = np.dot(S, A.T)
    # print(X.shape)
    
    A = np.array([[1,1,1],[0.5,2,1],[1.5,1,2]])
    X = np.dot(S, A.T)
    print(X.shape)
    
    # Step 5: Plot the mixed signals
    
    # TODO:
    # Create line plots for the mixed signals
    print("Mixed Signals")
    fig,axes = plt.subplots(3,1,figsize=(10,6), sharex=True)
    
    axes[0].plot(time,X[:,0])
    axes[0].set_title("Mixed Signal 1")
    
    axes[1].plot(time,X[:,1])
    axes[1].set_title("Mixed Signal 2")
    
    axes[2].plot(time,X[:,2])
    axes[2].set_title("Mixed Signal 3")
    
    plt.xlabel("Time")
    plt.show()
    
    # Step 6: Apply ICA
    
    # TODO:
    # ica = FastICA(...)
    # S_est = ...
    # A_est = ...
    # print(S_est.shape)
    # print(A_est.shape)
    
    ica = FastICA(n_components=3, random_state=42)
    S_est = ica.fit_transform(X)
    A_est = ica.mixing_
    print(S_est.shape)
    print(A_est.shape)
    
    # Step 7: Plot the recovered independent components
    
    # TODO:
    # Create line plots for the recovered signals
    # Compare visually with the original signals
    
    print("Recovered Signals")
    fig,axes = plt.subplots(3,1,figsize=(10,6), sharex=True)
    
    axes[0].plot(time,S_est[:,0])
    axes[0].set_title("Recovered Signal 1")
    
    axes[1].plot(time,S_est[:,1])
    axes[1].set_title("Recovered Signal 2")
    
    axes[2].plot(time,S_est[:,2])
    axes[2].set_title("Recovered Signal 3")
    
    fig,axes = plt.subplots(3,1,figsize=(10,6), sharex=True)
    
    for i in range(3):
        axes[i].plot(time, S[:,i], label="Original")
        axes[i].plot(time, S_est[:,i], label="Recovered", linestyle='--')
        axes[i].set_title(f"Signal {i+1}: Original vs Recovered")
        axes[i].legend()
    
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()
    
    # Optional Challenge
    # Modify the mixing matrix or add noise to the mixed signals,
    # then repeat the experiment and compare the output.
    
    # TODO:
    # 1. change A
    # 2. add noise
    # 3. run ICA again
    # 4. compare the results
    
    A_new = np.array([[1, 2, 1],
                      [2, 1, 1],
                      [1, 1, 2]])
    
    X_new = np.dot(S, A_new.T)
    
    
    #Step 2: Add Noise
    noise = 0.2 * np.random.normal(size=X_new.shape)
    X_noisy = X_new + noise
    
    ica = FastICA(n_components=3, random_state=42)
    S_new = ica.fit_transform(X_noisy)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(time, X_noisy[:,0])
    axes[0].set_title("Noisy Mixed Signal 1")
    
    axes[1].plot(time, X_noisy[:,1])
    axes[1].set_title("Noisy Mixed Signal 2")
    
    axes[2].plot(time, X_noisy[:,2])
    axes[2].set_title("Noisy Mixed Signal 3")
    
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()
    
    fig,axes = plt.subplots(3,1,figsize=(10,6), sharex=True)
    
    for i in range(3):
        axes[i].plot(time, S[:,i], label="Original")
        axes[i].plot(time, S_new[:,i], label="Recovered", linestyle='--')
        axes[i].set_title(f"Signal {i+1}: Original vs Recovered")
        axes[i].legend()
    
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


.. parsed-literal::

    (1000, 3)
    Original Source Signals
    


.. image:: output_1_1.png


.. parsed-literal::

    (1000, 3)
    Mixed Signals
    


.. image:: output_1_3.png


.. parsed-literal::

    (1000, 3)
    (3, 3)
    Recovered Signals
    


.. image:: output_1_5.png



.. image:: output_1_6.png



.. image:: output_1_7.png



.. image:: output_1_8.png


