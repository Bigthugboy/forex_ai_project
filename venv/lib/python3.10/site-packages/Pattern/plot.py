import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Set the style for the plots
sns.set_theme(style="whitegrid")

def plot_correlation_matrix(X):
	"""
	Plots a heatmap of the correlation matrix for the features in X.
	"""
	corr_matrix = np.corrcoef(X, rowvar=False)
	plt.figure(figsize=(10, 8))
	sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
	plt.title('Correlation Matrix Heatmap')
	plt.show()

def plot_pca(X, y):
	"""
	Plots the first two principal components of X, colored by the target variable y.
	"""
	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X)
	plt.figure(figsize=(8, 6))
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
	plt.title('PCA of the Dataset')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.colorbar(label='Class Label')
	plt.show()

def plot_tsne(X, y):
	"""
	Plots t-SNE visualization of X, colored by the target variable y.
	"""
	tsne = TSNE(n_components=2, random_state=42)
	X_tsne = tsne.fit_transform(X)
	plt.figure(figsize=(8, 6))
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
	plt.title('t-SNE of the Dataset')
	plt.xlabel('t-SNE Component 1')
	plt.ylabel('t-SNE Component 2')
	plt.colorbar(label='Class Label')
	plt.show()

def plot_kmeans(X, y, n_clusters=2):
	"""
	Plots KMeans clustering results on the first two principal components of X.
	"""
	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X)
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	y_kmeans = kmeans.fit_predict(X)
	plt.figure(figsize=(8, 6))
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.5)
	plt.title(f'KMeans Clustering (PCA) with {n_clusters} Clusters')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.colorbar(label='Cluster Label')
	plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
	"""
	Plots a confusion matrix for the true and predicted labels.
	"""
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
	plt.title('Confusion Matrix')
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	plt.show()

def plot_roc_curve(y_true, y_score):
	"""
	Plots the Receiver Operating Characteristic (ROC) curve.
	"""
	fpr, tpr, _ = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)
	plt.figure(figsize=(8, 6))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc='lower right')
	plt.show()

def plot_precision_recall_curve(y_true, y_score):
	"""
	Plots the Precision-Recall curve.
	"""
	precision, recall, _ = precision_recall_curve(y_true, y_score)
	plt.figure(figsize=(8, 6))
	plt.plot(recall, precision, color='b', lw=2)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.show()