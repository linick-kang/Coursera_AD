def merge_lane_lines(
        lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    
    ### START CODE HERE ### (≈ 25 lines in total)
    
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    
    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    
    # Step 2: Determine lines with slope less than horizontal slope threshold.
    non_hline = abs(slopes) >= min_slope_threshold
    slopes = slopes[non_hline]
    intercepts = intercepts[non_hline]
    nh_lines = lines[non_hline,:]
    
    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    line_class = np.ones(nh_lines.shape[0])*-1
    si_pair = []
    for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
        if len(si_pair) == 0:
            si_pair.append(np.array([slope, intercept]))
            line_class[0] = 0
        else:
            paired = False
            for j, (slope_tmp, intercept_tmp) in enumerate(si_pair):
                if abs(slope - slope_tmp) < slope_similarity_threshold or abs(intercept - intercept_tmp) < intercept_similarity_threshold:
                    paired = True
                    line_class[i] = j
                    break
            if paired == False:
                si_pair.append(np.array([slope, intercept]))
                line_class[i] = len(si_pair)
                           
    # Step 4: Merge all lines in clusters using mean averaging
    n_cluster = len(si_pair)
    merged_lines = np.zeros((0,4))
    for i in range(n_cluster):
        c_lines = nh_lines[(line_class == i),:]
        avg_lines = np.mean(c_lines, axis = 0)
        merged_lines = np.vstack((merged_lines, avg_lines))
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    ### END CODE HERE ###
    return merged_lines
