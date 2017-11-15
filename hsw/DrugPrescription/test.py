with open('test.csv', 'r') as f:
    f.readline()
    while True:
        temp = f.readline().split(',')
        if(len(temp) == 1): break
        print(temp)
    print('end')