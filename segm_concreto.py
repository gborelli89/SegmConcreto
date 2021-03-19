#--------------------------------------------------------------------------------
# Carregando as bibliotecas necessárias
#--------------------------------------------------------------------------------
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

#--------------------------------------------------------------------------------
# PARTE 1 - CARREGANDO A IMAGEM E APLICANDO CORREÇÃO DE CONTRASTE
#--------------------------------------------------------------------------------

# Leitura da imagem
im = cv2.imread('poros_crop1.tif')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # imagem em tons de cinza

# Aplicando correção de contraste
alpha = 1.8
beta = 0
imgrayC = cv2.convertScaleAbs(imgray, alpha=alpha, beta=beta)

#--------------------------------------------------------------------------------
# PARTE 2 - MODELO K-MEANS
#--------------------------------------------------------------------------------

# Transformando dados de tons de cinza em vetor para encontrar modelo K-Means
d = imgrayC.flatten()
d = np.reshape(d,(-1,1))

# Modelo com 3 clusters
kmeans = KMeans(3)
fitFun = kmeans.fit(d)
labels = fitFun.labels_
centers = fitFun.cluster_centers_
centers = centers.tolist()

# Aplicando mapa de cor aos clusters (apenas para visualização
L = [centers[i] for i in labels]
L = np.array(L, np.uint8)
L.shape = imgray.shape
L = cv2.applyColorMap(L, cv2.COLORMAP_RAINBOW)

cv2.imshow("Original", imgray)
cv2.imshow("Contraste", imgrayC)
cv2.imshow("KMeans", L)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--------------------------------------------------------------------------------
# PARTE 3 - DETERMINAÇÃO DO PERCENTUAL DE CADA FASE
#--------------------------------------------------------------------------------

# Ordenando clusters
ss_id = sorted(range(len(centers)), key=centers.__getitem__)
holes_id = centers.index(min(centers))
steel_id = centers.index(max(centers))
counts = Counter(labels)

# Determinando percentual de cada fase (para três fases)
total = 0
for i in counts:
	total = total + counts[i]

P_holes = 100*counts[holes_id]/total
P_steel = 100*counts[steel_id]/total
P_concrete = 100 - P_holes - P_steel
P = [P_holes, P_steel, P_concrete]
print(P)

#--------------------------------------------------------------------------------
# PARTE 4 - ENCONTRANDO OS CONTORNOS DE CADA FASE
#--------------------------------------------------------------------------------

# Valor 30 abaixo corresponde ao ponto médio entre o centro do
cluster do vazio e o cluster da matriz
_, thrHoles = cv2.threshold(L, 30, 255, cv2.THRESH_BINARY_INV)

# Valor 140 abaixo corresponde ao ponto médio entre o centro do
cluster da matriz e o cluster do reforço
_, thrSteel = cv2.threshold(L, 140, 255, cv2.THRESH_BINARY)
_,cntsHoles,_ = cv2.findContours(thrHoles, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
_,cntsSteel,_ = cv2.findContours(thrSteel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imContours = im.copy()

cv2.drawContours(imContours, cntsHoles, -1, (0,0,255), 1)
cv2.drawContours(imContours, cntsSteel, -1, (255,0,0), 1)
cv2.imshow("original", im)
cv2.imshow("holes", thrHoles)
cv2.imshow("steel", thrSteel)
cv2.waitKey(0)
cv2.destroyAllWindows()