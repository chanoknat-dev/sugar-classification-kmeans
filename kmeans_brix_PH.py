import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. อ่านข้อมูลจากไฟล์
df = pd.read_csv("kmeans_all.csv")

# 2. ลบคอลัมน์ที่ไม่จำเป็น
if 'Unnamed: 7' in df.columns:
    df = df.drop(columns=['Unnamed: 7'])

# 3. ลบแถวที่มีค่า NaN
df = df.dropna()

# 4. ตรวจสอบ DataFrame
if df.empty:
    print("DataFrame is empty after dropping NaN values.")
else:
    print("DataFrame is not empty, proceeding.")

# 5. เลือกคอลัมน์ที่สำคัญ
X = df[['pH', 'Brix']]

# 6. ปรับขนาดข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. กำหนดค่ากลาง (centroids)
initial_centroids = np.array([[6.42, 19.5],
                              [7.73, 26.5]])

# 8. ใช้ KMeans เพื่อจัดกลุ่ม
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 9. เพิ่ม labels กลับไปใน DataFrame
df['Cluster'] = labels

# 10. วิเคราะห์ผลลัพธ์
print(df)

# 11. สร้าง scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('K-means Clustering of Sugar Samples')
plt.xlabel('Scaled pH')
plt.ylabel('Scaled Brix')

# เพิ่ม legend เพื่อดูสีของแต่ละ cluster
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()
