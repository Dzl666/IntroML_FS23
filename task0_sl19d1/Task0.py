import numpy as np
from csv import reader, writer
import pandas as pd

def main():
    with open('train.csv', 'rt') as train_raw:
        readers = reader(train_raw, delimiter=',')
        # Id,y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
        x = list(readers)
        header = x[0]
        train_data = np.array(x[1:]).astype('float')
    with open('test.csv', 'rt') as test_raw:
        readers = reader(test_raw, delimiter=',')
        # Id, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
        x = list(readers)
        id_list = np.array(x[1:])[:,0]
        test_data = np.array(x[1:]).astype('float')

    predict = np.mean(test_data[:, 1:], axis=1)
    # with open('sample.csv', 'rt') as sample_raw:
    #     readers = reader(sample_raw, delimiter=',')
    #     # Id, y
    #     x = list(readers)
    #     header = x[0]
    #     y_gt = np.array(x[1:]).astype('float')

    df = pd.DataFrame({header[0]: id_list, header[1]: predict})
    df.to_csv('answer.csv', index=False, sep=',')

if __name__ == '__main__':
    main()