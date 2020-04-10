def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    
    ### START CODE HERE ### (≈ 23 lines in total)
    
    # Set thresholds:
    num_itr = 10  # RANSAC maximum number of iterations
    min_num_inliers = int(xyz_data.shape[1] * 0.5)  # RANSAC minimum number of inliers
    distance_threshold = 1  # Maximum distance from point to plane for point to be considered inlier
    n_inlier_max = 0
    
    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        # my method -> choose 3 points randomly which are far from each other with a distance threshold
        random_ok = False
        n = xyz_data.shape[1]
        while not random_ok:
            dbp_thres = 5 # distance between points threshold 5 m
            idx = np.random.choice(n, size=3, replace=False)
            points = xyz_data[:,idx]
            points_reorder = points[:,[1,2,0]]
            diff = points - points_reorder
            distance = np.linalg.norm(diff, axis=0)
            if sum(distance < dbp_thres) == 0:
                random_ok = True
        
        # Step 2: Compute plane model
        points = points.T
        XY_mat = np.ones((3,3))
        XY_mat[:,:2] = points[:,:2]
        Z_mat = points[:,2].reshape(3,1)
        #assume c = -1
        a, b, d = (np.linalg.inv(XY_mat) @ Z_mat).flatten()
        c = -1
        plane = np.array([a,b,c,d]).reshape(1,4)
        
        # Step 3: Find number of inliers
        pp = np.vstack((xyz_data, np.ones((1,n))))
        d2plane = abs(plane @ pp) / np.linalg.norm([a,b,c])
        n_inlier_cur = np.sum(d2plane < distance_threshold)
        
        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if n_inlier_cur > n_inlier_max:
            inlier_set = xyz_data[:,(d2plane < distance_threshold).ravel()]
            n_inlier_max = n_inlier_cur

        # Step 5: Check if stopping criterion is satisfied and break.         
        if n_inlier_max >= min_num_inliers:
            break
        
    # Step 6: Recompute the model parameters using largest inlier set.         
    xyz = inlier_set.T
    XY_set = np.hstack((xyz[:,:2], np.ones((n_inlier_max,1))))
    Z_set = xyz[:,2].reshape(-1,1)
    a_new, b_new, d_new = (np.linalg.inv(XY_set.T @ XY_set) @ XY_set.T @ Z_set).flatten()
    output_plane = np.array([a_new, b_new, -1, d_new])
    
    ### END CODE HERE ###
    
    return output_plane 
