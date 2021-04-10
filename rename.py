import os
path = "student"
f = os.listdir(path)
cnt = 0
for i in f:
    oldname = os.path.join(path, i)
    newname = os.path.join(path, "1_"+str(cnt)+".jpg")
    cnt = cnt+1
    os.rename(oldname,newname)