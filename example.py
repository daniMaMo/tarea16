import numpy as np
import pandas as pd

vectors = [(-1,-2), (1,1), (0,3), (3,2), (4,5), (2,6)]
newlist = []
VECTORS_SET =set(vectors)

for row in VECTORS_SET:
    for ROW in VECTORS_SET - {row}:
        if ((ROW[0] <= row[0]) and (ROW[1] <= row[1])):
            newlist.append(ROW)

DOMINADOS_SET = set(newlist)
print(DOMINADOS_SET)
RESULT = VECTORS_SET - DOMINADOS_SET
print(RESULT)
#RESULT_SET =
   # [ROW for ROW in vectors if ((ROW[0] < row[0]) and (ROW[1] < row[1]))]
    #print(row,newlist)
    #for ROW in vectors:
        #newlist = [ROW for ROW in vectors if not (row[0] >= ROW[0]) and (row[1] >= ROW[1])]
        #if not ((row[0] >= ROW[0]) and (row[1] >= ROW[1])):
            #RESULT.append(row)
            #print('paso', RESULT)

