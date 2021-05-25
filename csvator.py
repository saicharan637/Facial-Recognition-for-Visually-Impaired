import csv
import numpy as np

def read_from_csv(pathh):
    embds = []
    clss = []
    with open(pathh,'r',encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row) == 0):
                continue
            temp = np.array(row[:-1])
            embds.append(temp.astype(np.float).tolist())
            clss.append(row[-1])
    return embds, clss

if __name__ == "__main__":
    a,b = read_from_csv('trainedData\\f1.csv')
    print(type(a[0][0]))
    print(a[0])