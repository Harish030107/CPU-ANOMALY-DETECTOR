import joblib
import json
import numpy as np
import os

def extract_tree_structure(tree):
    """Extract decision tree structure from sklearn tree"""
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    
    nodes = []
    for i in range(n_nodes):
        node = {
            'id': int(i),
            'feature': int(feature[i]) if feature[i] >= 0 else -1,
            'threshold': float(threshold[i]) if feature[i] >= 0 else 0.0,
            'left_child': int(children_left[i]) if children_left[i] != -1 else -1,
            'right_child': int(children_right[i]) if children_right[i] != -1 else -1,
            'value': value[i].tolist(),
            'is_leaf': bool(children_left[i] == -1)
        }
        nodes.append(node)
    
    return nodes

def convert_rf_to_json(pkl_path, json_path):
    """Convert sklearn RandomForest pickle to JSON"""
    
    if not os.path.exists(pkl_path):
        print(f"[ERROR] File not found: {pkl_path}")
        return False
    
    print(f"[INFO] Loading model from {pkl_path}")
    rf_model = joblib.load(pkl_path)
    
    # Extract model information
    model_data = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': rf_model.n_estimators,
        'n_features': rf_model.n_features_in_,
        'n_classes': int(rf_model.n_classes_),
        'classes': rf_model.classes_.tolist(),
        'trees': []
    }
    
    # Extract each tree
    print(f"[INFO] Extracting {rf_model.n_estimators} trees...")
    for idx, estimator in enumerate(rf_model.estimators_):
        tree_data = {
            'tree_id': idx,
            'nodes': extract_tree_structure(estimator.tree_)
        }
        model_data['trees'].append(tree_data)
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{rf_model.n_estimators} trees")
    
    # Save to JSON
    print(f"[INFO] Saving to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"[OK] Model converted successfully!")
    print(f"  - Trees: {model_data['n_estimators']}")
    print(f"  - Features: {model_data['n_features']}")
    print(f"  - Classes: {model_data['classes']}")
    
    return True

if __name__ == '__main__':
    pkl_file = 'rf_labeled_model.pkl'
    json_file = 'rf_freq_model.json'
    
    convert_rf_to_json(pkl_file, json_file)
