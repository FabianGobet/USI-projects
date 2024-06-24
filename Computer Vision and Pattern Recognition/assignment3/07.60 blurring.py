# Set the stage
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import math

# Read and view an image
A = cv2.imread("watch.pgm", cv2.IMREAD_GRAYSCALE)
cv2_imshow(A)
M, N = A.shape

#
# box filtering
#
a = 1
b = a
m = 2*a+1
n = 2*b+1

B = np.copy(A)
w = 1/(m*m)
for i in range(a,M-a):
  for j in range(b,N-b):
    I = 0
    for s in range(-a,a+1):
      for t in range(-b,b+1):
        I = I + w*A[i+s,j+t]
    B[i,j] = round(I)

cv2_imshow(B)

#
# Gaussian filtering
#
a = 1
b = a
m = 2*a+1
n = 2*b+1

sigma = 1
g = np.zeros((m,m))
for s in range(-a,a+1):
  for t in range(-b,b+1):
    r2 = s*s + t*t
    g[s+a,t+b] = math.exp(-r2/2/(sigma*sigma))
w = g/np.sum(g)
print(w)
    
B = np.copy(A)
for i in range(a,M-a):
  for j in range(b,N-b):
    I = 0
    for s in range(-a,a+1):
      for t in range(-b,b+1):
        I = I + w[s+a,t+b] * A[i+s,j+t]
    B[i,j] = round(I)

cv2_imshow(B)

#
# Gaussian filtering
# using separability of the filter
# with replicate padding
#
a = 1
b = a
m = 2*a+1
n = 2*b+1

sigma = 1
g = np.zeros((m))
for s in range(-a,a+1):
  g[s+a] = math.exp(-s*s/2/(sigma*sigma))
c = g/np.sum(g)
r = c

A1 = np.copy(A)
for i in range(0,M):
  for j in range(0,N):
    I = 0
    for s in range(-a,a+1):
      if (i+s < 0):
        I = I + c[s+a] * A[0,j]
      elif (i+s > M-1):
        I = I + c[s+a] * A[M-1,j]
      else:
        I = I + c[s+a] * A[i+s,j]
    A1[i,j] = round(I)

B1 = np.copy(A1)
for i in range(0,M):
  for j in range(0,N):
    I = 0
    for t in range(-b,b+1):
      if (j+t < 0):
        I = I + r[t+b] * A1[i,0]
      elif (j+t > N-1):
        I = I + r[t+b] * A1[i,N-1]
      else:
        I = I + r[t+b] * A1[i,j+t]
    B1[i,j] = round(I)

cv2_imshow(B1)

print(w)
print(np.matmul(np.transpose(np.asmatrix(c)),np.asmatrix(r)))

cv2_imshow(abs(B-B1))
cv2_imshow(abs(B-B1.astype(np.float32)))

np.linalg.matrix_rank(w)

