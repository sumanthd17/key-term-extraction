file = "stopwords.txt"
stoplist = []
with open(file) as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    for line in lines:
        stoplist.append(line)

print(stoplist)
