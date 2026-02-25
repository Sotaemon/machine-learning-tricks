def main():
    feat_ranges = {}
    const_feat = ['Age', 'Fare']
    bins = 10

    data = pd.read_csv('./data/titanic/train.csv')
    print(data.info())
    print(data[:5])

    data.drop(columns = ['PassengerId', 'Name', 'Ticket'], inplace = True)
    
    for feat in const_feat:
        min_val = np.nanmin(data[feat])
        max_val = np.nanmax(data[feat])
        feat_ranges[feat] = np.linspace(min_val, max_val, bins).tolist()

if __name__ == "__main__":
    main()