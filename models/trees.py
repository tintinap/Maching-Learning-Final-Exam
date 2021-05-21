import numpy as np
import math

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value

class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, row, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = row[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(row, tree.left)
        else:
            return self.make_prediction(row, tree.right)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(row, self.root) for row in X]
        return preditions

def random_forest_regression(X, y, test_X, n_trees, n_features, sample_sz):
    np.random.seed(1)
    #row_idx, col_idx
    rf_row_4_each_tree = [(np.random.permutation(len(y))[:sample_sz], np.random.permutation(X.shape[1])[:n_features]) for i in range(n_trees)]
    # example
    #     [(array([13525, 10012,  9179, ...,  3144,   688,  2468]), array([6, 7, 5, 0])),
    #     (array([ 6990,  9903,  7602, ...,  6707,  1525, 11287]), array([5, 4, 1, 3])),
    #     (array([3844, 6751, 1905, ..., 7591, 7088, 2528]), array([7, 0, 4, 3])),
    #     (array([4287, 4004, 3043, ..., 6780, 9880, 9819]), array([2, 1, 7, 5])),
    #     (array([12379,  1106,  5205, ...,  8698, 11813,  2375]), array([3, 7, 6, 4])),
    #     (array([13667,  4252, 14082, ...,  1114,  7080, 10546]), array([2, 3, 6, 1])),
    #     (array([ 8261,  1926, 11879, ...,  4587,  6752,  9817]), array([6, 2, 5, 4])),
    #     (array([ 4129,  4944, 14082, ..., 14875,  6144,  8428]), array([5, 6, 3, 7])),
    #     (array([10431,   166,  5210, ..., 13999,  5311, 14215]), array([0, 3, 1, 4])),
    #     (array([13519,  4921,  7531, ..., 14348,  8880,  8521]), array([7, 0, 5, 3]))]
    trees = list()
    for i in range(n_trees):
        dt_reg = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
        dt_reg.fit(
              X.iloc[:, rf_row_4_each_tree[i][1]].values,
              y.iloc[rf_row_4_each_tree[i][0]].values.reshape(-1,1)
        )
        trees.append(dt_reg)
    prediction = [tree.predict(test_X.values) for tree in trees]

    print('for_checking')
    for p in prediction:
        print(p[:5])

    return np.mean(prediction, axis=0).tolist()

# class RandomForestRegressor():
#     def __init__(self, X, y, n_trees, n_features, sample_sz, max_depth=10, min_samples_split=5):
#         np.random.seed(12)
#         if n_features == 'sqrt':
#             self.n_features = int(np.sqrt(X.shape[1]))
#         elif n_features == 'log2':
#             self.n_features = int(np.log2(X.shape[1]))
#         else:
#             self.n_features = n_features
#         print(self.n_features, "sha: ",X.shape[1])    
#         self.X, self.y, self.sample_sz, self.max_depth, self.min_samples_split  = X, y, sample_sz, max_depth, min_samples_split
#         print('creating_trees')
#         self.trees = [self.create_tree() for i in range(n_trees)]

#     def create_tree(self):
#         idxs = np.random.permutation(len(self.y))[:self.sample_sz]
#         f_idxs = np.random.permutation(self.X.shape[1])[:self.n_features]
#         print(f_idxs)
#         print(len(idxs))
#         dt_regressor = DecisionTreeRegressor(
#             min_samples_split=self.min_samples_split, 
#             max_depth=self.max_depth
#         )
#         print('b4 fit')
#         return dt_regressor.fit(
#                 self.X.iloc[idxs].iloc[:,f_idxs].values, self.y.iloc[idxs].values.reshape(-1,1)
#             )

#     def make_prediction(self, row, trees):
#         ''' function to predict a single data point '''

#         return np.mean([tree.make_prediction(row, tree.root) for tree in trees])

#     def predict(self, X):
#         ''' function to predict new dataset '''

#         preditions = [self.make_prediction(row, self.trees) for row in X]
#         return preditions


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)