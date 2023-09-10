# first line: 11
@mem.cache
def dataLoading(path, byte_num=280):
    # loading data
    x=[]
    labels=[]
    with (open(path,'r') ) as data_from:
        csv_reader=csv . reader(data_from)
        byte_num=csv_reader.line_num
        for i in csv_reader:
            x.append(i[0:byte_num])
            labels.append(i[byte_num])

    for i in range(len(x)):
        for j in range(byte_num):
            x[i][j] = float(x[i][j])
    for i in range(len(labels)):
        labels[i] = float(labels[i])
    x = np.array(x,dtype=np.float32)
    labels = np.array(labels)
    return x,labels
