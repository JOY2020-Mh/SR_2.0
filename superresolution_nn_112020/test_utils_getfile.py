import os

#def get_files(path):
    # read a folder, return the image name

path = './HR'
ret = []
a = []
for root, dirs, files in os.walk(path):
    #print(root)
    #print(dirs)
    #print(files)

    for filespath in files:
        #print(filespath)
        ret.append(os.path.join(path, filespath))
    #return ret
    #print(ret)
    #print(len(ret))


a.extend(ret)
print(a)
