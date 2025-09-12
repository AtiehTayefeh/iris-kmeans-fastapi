import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# 1. بارگذاری مجموعه داده Iris
iris = load_iris()
X = iris.data
y = iris.target # برچسب‌های واقعی گونه‌ها (برای مقایسه، نه برای خود K-Means)
feature_names = iris.feature_names
target_names = iris.target_names

# 2. انتخاب دو ویژگی برای رسم نمودار (مثلاً طول کاسبرگ و عرض کاسبرگ)
# ویژگی 0: Sepal Length, ویژگی 1: Sepal Width
X_subset = X[:, [0, 1]]

# 3. اجرای الگوریتم K-Means
# ما می‌دانیم که 3 گونه داریم، پس K=3 را انتخاب می‌کنیم
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # n_init=10 برای اجرای ۱۰باره با مقادیر اولیه مختلف و انتخاب بهترین نتیجه
kmeans.fit(X_subset)

# گرفتن برچسب‌های خوشه‌ها
cluster_labels = kmeans.labels_

# گرفتن مراکز خوشه‌ها
cluster_centers = kmeans.cluster_centers_

# 4. رسم نمودار با استفاده از Seaborn

# بهتر است داده‌ها را در یک DataFrame قرار دهیم تا کار با Seaborn راحت‌تر باشد
df = pd.DataFrame(X_subset, columns=[feature_names[0], feature_names[1]])
df['Cluster'] = cluster_labels # اضافه کردن ستون خوشه‌ها

plt.figure(figsize=(10, 7))

# استفاده از Seaborn برای رسم Scatter Plot با رنگ‌بندی خوشه‌ها
# hue='Cluster' باعث می‌شود نقاط بر اساس ستون 'Cluster' رنگ‌آمیزی شوند.
sns.scatterplot(data=df, x=feature_names[0], y=feature_names[1], hue='Cluster', palette='viridis', s=80, alpha=0.8)

# رسم مراکز خوشه‌ها
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='مراکز خوشه‌ها')

plt.title('خوشه‌بندی داده‌های Iris با K-Means (طول و عرض کاسبرگ)')
plt.xlabel(feature_names[0]) # Sepal Length
plt.ylabel(feature_names[1]) # Sepal Width
plt.legend()
plt.grid(True)
plt.show()

# برای مقایسه، می‌توانیم یک نمودار دیگر با استفاده از برچسب‌های واقعی گونه‌ها نیز رسم کنیم:
df['Species'] = iris.target
df['Species'] = df['Species'].map({i: target_names[i] for i in range(len(target_names))})

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x=feature_names[0], y=feature_names[1], hue='Species', palette='tab10', s=80, alpha=0.8)
plt.title('گونه‌های واقعی گل Iris (طول و عرض کاسبرگ)')
plt.xlabel(feature_names[0]) # Sepal Length
plt.ylabel(feature_names[1]) # Sepal Width
plt.legend()
plt.grid(True)
plt.show()

print("\n--- مقایسه نتایج K-Means با گونه‌های واقعی ---")
# برای مقایسه دقیق‌تر، ماتریس درهم‌ریختگی (Confusion Matrix) یا معیارهای دیگر مانند Adjusted Rand Index مناسب هستند.
# اما اینجا یک نمای کلی با ترکیب داده‌ها نشان می‌دهیم:
df['KMeans_Cluster'] = cluster_labels
print(df.head())
