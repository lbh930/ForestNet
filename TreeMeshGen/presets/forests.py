from presets.trees import broadleaf_config, oak_config, pine_config

broadleaf_forest_config = {
    "tree_config": broadleaf_config,
    "width": 50, #in meters, range on x axis
    "length": 50, #in meters, range on y axis
    "average_height": 8,
    "variance_height": 1, #height randomized based on normal distribution"
    "tree_count": 50,
    "minimal_distance": 5, # minimal distance between distinct trees.
}

# Mimicking the forest from L1W point cloud of TreeLearn paper
L1W_forest_config = { 
    "tree_config": broadleaf_config,
    "width": 112, #in meters, range on x axis
    "length": 103, #in meters, range on y axis
    "average_height": 27,
    "variance_height": 6.5, #height randomized based on normal distribution"
    "tree_count": 156,
    "minimal_distance": 4.5, # minimal distance between distinct trees.
}

test_forest_config = { 
    "tree_config": broadleaf_config,
    "width": 100, 
    "length": 100,
    "average_height": 30,
    "variance_height": 6.5,
    "tree_count": 150,
    "minimal_distance": 4.5,
}