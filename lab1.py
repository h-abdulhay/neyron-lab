import numpy as np

print("=" * 40)
print("1. PERCEPTRON")
print("=" * 40)

x1, x2 = 1.0, 0.5
w1, w2 = 0.8, -0.5
b = 0.3

z = x1*w1 + x2*w2 + b
print(f"z = {z:.4f}")

print("=" * 40)
print("2. SIGMOID va RELU")
print("=" * 40)

sigmoid = 1 / (1 + np.exp(-z))
relu = max(0, z)
print(f"Sigmoid: {sigmoid:.4f}")
print(f"ReLU:    {relu:.4f}")

print("=" * 40)
print("3. FORWARD PROPAGATION")
print("=" * 40)

W1 = np.array([[0.4, -0.3],
               [0.2,  0.7]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.6, 0.5])
b2 = 0.3

x = np.array([x1, x2])
z1 = np.dot(W1, x) + b1
h  = 1 / (1 + np.exp(-z1))
z2 = np.dot(W2, h) + b2
y_hat = 1 / (1 + np.exp(-z2))
print(f"1-qavat: {h}")
print(f"Chiqish: {y_hat:.4f}")

print("=" * 40)
print("4. BACKPROPAGATION")
print("=" * 40)

y_true = 1.0
lr = 0.1
loss = -(y_true * np.log(y_hat) + (1-y_true) * np.log(1-y_hat))
dL = -(y_true/y_hat - (1-y_true)/(1-y_hat))
dsig = y_hat * (1 - y_hat)
dz = dL * dsig
w_old = 0.6
w_new = w_old - lr * dz
print(f"Loss:      {loss:.4f}")
print(f"Gradient:  {dz:.4f}")
print(f"w_eski:    {w_old:.4f}")
print(f"w_yangi:   {w_new:.4f}")

print("=" * 40)
print("5. CNN va RNN farqi")
print("=" * 40)
print("CNN - rasmlar uchun, piksellarni tahlil qiladi")
print("RNN - matn/ovoz uchun, ketma-ketlikni eslab qoladi")

print("=" * 40)
print("6. TOKENIZATSIYA")
print("=" * 40)
matn = "Kiberhujum aniqlandi tizimda"
tokenlar = matn.lower().split()
print(f"Matn: {matn}")
print(f"Tokenlar: {tokenlar}")
print(f"Soni: {len(tokenlar)}")

print("=" * 40)
print("7. KIBERXAVFSIZLIK")
print("=" * 40)
trafiklar = [0.2, 0.15, 0.9, 0.85, 0.1]
chegara = 0.7
for i, t in enumerate(trafiklar):
    holat = "HUJUM" if t > chegara else "normal"
    print(f"Paket {i+1}: {t} -> {holat}")