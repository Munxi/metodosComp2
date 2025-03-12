import re
import numpy as np
import numba as nb

f = open('mobyDick.txt','r')
s = f.read()
s = re.sub(r".*CHAPTER 1\. Loomings\.\n\n",'',s,1,flags=re.DOTALL)
s = re.sub(r"\n\n\nEpilogue.*",'',s,1,flags=re.DOTALL)
s = re.sub(r"\n\nCHAPTER [0-9]+[^\n]*\n",'',s)
s = re.sub(r'[—-]', ' ', s)
s = re.sub(r'\[[^\]]*\]', '', s)
s = re.sub(r' {2,}', ' ', s.lower())
s = s.replace("\r\n","\n").replace("\n\n","#").replace("\n"," ").replace("#","\n\n")
s = re.sub(r'[^a-z\s\.\,\;\:\!\?\n]', '', s)

f.close()
f = open('formateado.txt','w')
f.write(s)
f.close()

mapping = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
    't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
    ' ': 27, '.': 28, ',': 29, ';': 30, ':': 31, '!': 32, '?': 33, '\n': 34
}

combinat = {}
n = 7
def entrenamiento(n,combinat,s,mapping):
    j = 0
    for i in range(n,len(s)):
        if s[i-n:i] in combinat:
            combinat[s[i-n:i]][mapping[s[i]]-1] += 1
        else:
            combinat[s[i-n:i]] = np.zeros(34,dtype=np.int32)
            combinat[s[i-n:i]][mapping[s[i]]-1] += 1
        if j==1178505:
            None
        j+=1
    return j

max = entrenamiento(n,combinat,s,mapping)