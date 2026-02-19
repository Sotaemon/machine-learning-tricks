from sklearn.datasets import make_classification

class RandomForest():
    pass

def main():
    features, lables = make_classification(
        n_samples = 1000,
        n_features = 16,
        n_informative = 5,
        n_redundant = 2,
        n_classes = 2,
        flip_y = 0.1,
        random_state = 0
    )
    print(features.shape)





if __name__ == '__main__':
    main()