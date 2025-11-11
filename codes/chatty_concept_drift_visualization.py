# import numpy as np
# import matplotlib.pyplot as plt

# # --- Simulated Data ---
# np.random.seed(42)
# n_samples = 300
# drift_point = 150

# # Two Gaussian blobs (same distribution before and after drift)
# X_left = np.random.normal(loc=[2, 5], scale=1.0, size=(n_samples // 2, 2))
# X_right = np.random.normal(loc=[6, 5], scale=1.0, size=(n_samples // 2, 2))
# X = np.vstack([X_left, X_right])

# # Labels before and after drift
# y_before = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))  # left=0, right=1
# y_after = 1 - y_before  # flip labels

# # Stream: first half before drift, second half after drift
# X_stream = np.vstack([X, X])
# y_stream = np.hstack([y_before, y_after])

# # --- Combined Figure ---
# fig = plt.figure(figsize=(12, 8))

# # Scatter Before Drift
# ax1 = plt.subplot(2, 2, 1)
# ax1.scatter(X[:, 0], X[:, 1], c=y_before, cmap="bwr", edgecolor="k")
# ax1.axvline(4, color="black", linestyle="--", label="Decision Boundary")
# ax1.set_title("Before Drift\n(Left=0, Right=1)")
# ax1.legend()

# # Scatter After Drift
# ax2 = plt.subplot(2, 2, 2)
# ax2.scatter(X[:, 0], X[:, 1], c=y_after, cmap="bwr", edgecolor="k")
# ax2.axvline(4, color="black", linestyle="--")
# ax2.set_title("After Drift\n(Left=1, Right=0)")

# # Stream Feature
# ax3 = plt.subplot(2, 1, 2)
# ax3.plot(X_stream[:, 0], "b.", alpha=0.5, label="Feature 1")
# ax3.plot(y_stream * max(X_stream[:, 0]), "ro", alpha=0.7, label="Class (scaled)")
# ax3.axvline(n_samples, color="black", linestyle="--", label="Concept Drift")
# ax3.set_ylabel("Value / Class")
# ax3.set_xlabel("Index")
# ax3.legend()

# plt.tight_layout()
# plt.show()


##########

# import numpy as np
# import matplotlib.pyplot as plt

# # --- Simulated Data ---
# np.random.seed(42)
# n_samples = 600
# drift_points = [200, 400]  # where concept drift happens

# # Generate 3 feature streams
# X = np.random.normal(loc=0, scale=1, size=(n_samples, 3))

# # Labels with concept drift (3 rules)
# y = np.zeros(n_samples, dtype=int)

# # Concept 1: class = 1 if feature 0 > 0
# y[:drift_points[0]] = (X[:drift_points[0], 0] > 0).astype(int)

# # Concept 2: class = 1 if feature 1 > 0
# y[drift_points[0]:drift_points[1]] = (X[drift_points[0]:drift_points[1], 1] > 0).astype(int)

# # Concept 3: class = 1 if feature 0 + feature 2 > 0
# y[drift_points[1]:] = (X[drift_points[1]:, 0] + X[drift_points[1]:, 2] > 0).astype(int)

# # --- Plotting ---
# fig = plt.figure(figsize=(15, 10))

# # Scatter plots for each concept
# ax1 = plt.subplot(2, 3, 1)
# ax1.scatter(X[:drift_points[0], 0], X[:drift_points[0], 1],
#             c=y[:drift_points[0]], cmap="bwr", edgecolor="k")
# ax1.set_title("Concept 1:\nclass = (Feature 0 > 0)")

# ax2 = plt.subplot(2, 3, 2)
# ax2.scatter(X[drift_points[0]:drift_points[1], 0], X[drift_points[0]:drift_points[1], 1],
#             c=y[drift_points[0]:drift_points[1]], cmap="bwr", edgecolor="k")
# ax2.set_title("Concept 2:\nclass = (Feature 1 > 0)")

# ax3 = plt.subplot(2, 3, 3)
# ax3.scatter(X[drift_points[1]:, 0], X[drift_points[1]:, 1],
#             c=y[drift_points[1]:], cmap="bwr", edgecolor="k")
# ax3.set_title("Concept 3:\nclass = (Feature 0 + Feature 2 > 0)")

# # Feature streams
# ax4 = plt.subplot(4, 1, 3)
# for i in range(3):
#     ax4.plot(X[:, i], label=f"Feature {i}")
# for dp in drift_points:
#     ax4.axvline(dp, color="black", linestyle="--")
# ax4.set_ylabel("Feature values")
# ax4.legend()

# # Class stream
# ax5 = plt.subplot(4, 1, 4, sharex=ax4)
# ax5.plot(y, "ro", alpha=0.6)
# for dp in drift_points:
#     ax5.axvline(dp, color="black", linestyle="--", label="Concept Drift")
# ax5.set_ylabel("Class")
# ax5.set_xlabel("Index")
# ax5.legend()

# plt.tight_layout()
# plt.show()

##########


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Generate original dataset (concept 1)
X1, y1 = make_classification(n_samples=500, n_features=2, n_informative=2,
                             n_redundant=0, n_clusters_per_class=1,
                             class_sep=1.5, flip_y=0.01, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Generate drift dataset (concept 2)
X2, y2 = make_classification(n_samples=500, n_features=2, n_informative=2,
                             n_redundant=0, n_clusters_per_class=2,
                             class_sep=0.7, flip_y=0.10, random_state=12)

# Train on first distribution
clf = RandomForestClassifier(random_state=42)
clf.fit(X1_train, y1_train)

# Evaluate
acc1 = accuracy_score(y1_test, clf.predict(X1_test))
acc2 = accuracy_score(y2, clf.predict(X2))

# Decision boundary plotting function
def plot_decision_boundary(X, y, clf, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7,5))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    return plt


# Plotting both
plt1 = plot_decision_boundary(X1_test, y1_test, clf,
    f"Concept 1 (Original)\nAccuracy: {acc1:.2f}")
plt1.savefig("decision_boundary_original.png")
plt2 = plot_decision_boundary(X2, y2, clf,
    f"Concept 2 (Drift)\nAccuracy: {acc2:.2f}")
plt2.savefig("decision_boundary_drift.png")

