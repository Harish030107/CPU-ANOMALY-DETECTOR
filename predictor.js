import fs from 'fs';

/**
 * Random Forest Predictor for converted sklearn models
 */
export class RandomForestPredictor {
    constructor(modelData) {
        this.modelData = modelData;
        this.nEstimators = modelData.n_estimators;
        this.nFeatures = modelData.n_features;
        this.classes = modelData.classes;
        this.trees = modelData.trees;
    }

    /**
     * Predict class for a single sample using a decision tree
     */
    predictTree(treeData, features) {
        let nodeId = 0;
        const nodes = treeData.nodes;

        while (true) {
            const node = nodes[nodeId];

            if (node.is_leaf) {
                // Return the class with highest value
                const values = node.value[0];
                return values.indexOf(Math.max(...values));
            }

            // Navigate tree based on feature threshold
            const featureValue = features[node.feature];
            if (featureValue <= node.threshold) {
                nodeId = node.left_child;
            } else {
                nodeId = node.right_child;
            }
        }
    }

    /**
     * Predict class for features using all trees (majority vote)
     */
    predict(features) {
        if (features.length !== this.nFeatures) {
            throw new Error(`Expected ${this.nFeatures} features, got ${features.length}`);
        }

        // Get predictions from all trees
        const votes = new Array(this.classes.length).fill(0);

        for (const tree of this.trees) {
            const prediction = this.predictTree(tree, features);
            votes[prediction]++;
        }

        // Return class with most votes
        const maxVotes = Math.max(...votes);
        return this.classes[votes.indexOf(maxVotes)];
    }

    /**
     * Predict with probability scores
     */
    predictProba(features) {
        if (features.length !== this.nFeatures) {
            throw new Error(`Expected ${this.nFeatures} features, got ${features.length}`);
        }

        const votes = new Array(this.classes.length).fill(0);

        for (const tree of this.trees) {
            const prediction = this.predictTree(tree, features);
            votes[prediction]++;
        }

        // Convert to probabilities
        const total = votes.reduce((a, b) => a + b, 0);
        const probabilities = votes.map(v => v / total);

        return {
            classes: this.classes,
            probabilities: probabilities,
            prediction: this.classes[probabilities.indexOf(Math.max(...probabilities))]
        };
    }

    /**
     * Load model from JSON file
     */
    static load(jsonPath) {
        const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        return new RandomForestPredictor(data);
    }
}
