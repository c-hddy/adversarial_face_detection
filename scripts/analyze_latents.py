import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap # UMAP 라이브러리 (설치 필요: pip install umap-learn)
import pandas as pd # 데이터를 다루기 위해 pandas 사용
import seaborn as sns # 시각화를 위해 seaborn 사용
import argparse
import sys

# common 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_latent_dir # common/utils.py

# --- 1. 명령줄 인수 파싱 ---
parser = argparse.ArgumentParser(description='Latent Space Visualization and Quantitative Analysis')
parser.add_argument('--ri_latent_dir', type=str, default='data/latents/RI/', 
                    help='Path to the directory containing RI latent vectors (.npy files). Defaults to data/latents/RI/.')
parser.add_argument('--acai_latent_dir', type=str, default='data/latents/ACAI/', 
                    help='Path to the directory containing ACAI latent vectors (.npy files). Defaults to data/latents/ACAI/.')
parser.add_argument('--file_count_per_class', type=int, default=100, 
                    help='Number of latent vector files to load from each class for analysis. Defaults to 100.')
parser.add_argument('--output_plot_dir', type=str, default='results/latent_visualization/', 
                    help='Path to save the visualization plots. Defaults to results/latent_visualization/.')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for reproducibility of t-SNE and UMAP. Defaults to 42.')

# --- 2. 잠재 벡터 로드 함수 (이전과 동일) ---
# common/utils.py에서 임포트하여 사용

# --- 3. 정량적 분석 함수 (2클래스에 맞게 출력 변경) ---
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity 

def analyze_latent_distributions(class_data_dict: dict):
    """
    클래스별 잠재 벡터 분포에 대한 정량적 분석을 수행합니다 (RI, ACAI 두 클래스).

    Args:
        class_data_dict (dict): {클래스명: 스케일링된 잠재 벡터 배열} 형태의 딕셔너리.
                                (예: {'RI': ri_scaled_data, 'ACAI': acai_scaled_data})

    Returns:
        tuple: (class_means, class_variances, l2_distances, avg_cosine_similarities) 튜플.
    """
    class_means = {}
    class_variances = {}
    
    print("\n--- Class Statistics ---")
    for name, data in class_data_dict.items():
        if len(data) == 0:
            print(f"  {name}: No data available.")
            continue
        
        mean_vector = np.mean(data, axis=0)
        variance = np.var(data) 
        
        class_means[name] = mean_vector
        class_variances[name] = variance
        print(f"  {name}: Mean Vector (first 5 dims): {mean_vector[:5].round(4)}, Variance: {variance:.7f}")

    print("\n--- Distance between Class Centers (L2 Norm) ---")
    class_names = list(class_data_dict.keys())
    l2_distances = {}
    # RI vs ACAI만 계산 (두 클래스만 존재하므로)
    if 'RI' in class_means and 'ACAI' in class_means:
        dist = np.linalg.norm(class_means['RI'] - class_means['ACAI'])
        l2_distances["RI vs ACAI"] = dist
        print(f"  RI vs ACAI: {dist:.4f}")
    else:
        print("  Not enough classes (RI and ACAI) to calculate L2 distance.")
    
    print("\n--- Average Cosine Similarity to Class Center ---")
    avg_cosine_similarities = {}
    for name, data in class_data_dict.items():
        if len(data) == 0:
            continue
        
        center = class_means[name].reshape(1, -1)
        similarities = cosine_similarity(data, center).flatten()
        avg_sim = np.mean(similarities)
        avg_cosine_similarities[name] = avg_sim
        print(f"  {name}: {avg_sim:.4f}")

    return class_means, class_variances, l2_distances, avg_cosine_similarities


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    args = parser.parse_args()

    # 데이터 로드 및 레이블링 (RI, ACAI만 로드)
    print("Loading latent data...")
    ri_latents = load_latent_dir(args.ri_latent_dir, count=args.file_count_per_class)
    ri_labels = np.array(['RI'] * len(ri_latents))

    acai_latents = load_latent_dir(args.acai_latent_dir, count=args.file_count_per_class)
    acai_labels = np.array(['ACAI'] * len(acai_latents))

    all_latents = np.vstack([ri_latents, acai_latents]) # SI 제거
    all_labels = np.concatenate([ri_labels, acai_labels]) # SI 제거

    print(f"Total samples loaded for analysis: {len(all_latents)}")
    if len(all_latents) > 0:
        print(f"Latent vector dimension: {all_latents.shape[1]}")
    else:
        print("No latent vectors loaded. Exiting.")
        sys.exit()

    # 데이터 정규화 (StandardScaler)
    scaler = StandardScaler()
    all_latents_scaled = scaler.fit_transform(all_latents)
    print("Latent vectors standardized.")

    # 정량적 분석 실행
    class_data_for_analysis = {
        'RI': all_latents_scaled[all_labels == 'RI'],
        'ACAI': all_latents_scaled[all_labels == 'ACAI'] # SI 제거
    }
    means, variances, l2_distances, cosine_similarities = analyze_latent_distributions(class_data_for_analysis)

    print("\n--- Quantitative Analysis Complete ---")
    print("Full Class Variances:")
    for name, var in variances.items():
        print(f"  {name}: {var:.7f}")

    print("\nFull L2 Distances between Class Centers:")
    for pair, dist in l2_distances.items():
        print(f"  {pair}: {dist:.4f}")

    print("\nFull Average Cosine Similarities to Class Center:")
    for name, sim in cosine_similarities.items():
        print(f"  {name}: {sim:.4f}")


    # 차원 축소 (PCA, t-SNE, UMAP)
    print("\nPerforming dimension reduction...")
    # PCA
    pca = PCA(n_components=2, random_state=args.seed)
    pca_results = pca.fit_transform(all_latents_scaled)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")

    # t-SNE
    print("Performing t-SNE (this may take a while for larger datasets)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=args.seed, n_jobs=-1)
    tsne_results = tsne.fit_transform(all_latents_scaled)

    # UMAP
    print("Performing UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=args.seed)
    umap_results = reducer.fit_transform(all_latents_scaled)

    # 결과 시각화 (2클래스에 맞게)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Latent Space Visualization (Count per class: {args.file_count_per_class}, Classes: RI, ACAI)', fontsize=16) # 제목 변경

    df = pd.DataFrame({
        'component_1': pca_results[:, 0],
        'component_2': pca_results[:, 1],
        'label': all_labels
    })
    # palette는 2개의 클래스에 맞게 자동으로 조정됨
    sns.scatterplot(x='component_1', y='component_2', hue='label', palette='viridis', ax=axes[0], data=df, s=50, alpha=0.7)
    axes[0].set_title('PCA')
    axes[0].grid(True)

    df_tsne = pd.DataFrame({
        'component_1': tsne_results[:, 0],
        'component_2': tsne_results[:, 1],
        'label': all_labels
    })
    sns.scatterplot(x='component_1', y='component_2', hue='label', palette='viridis', ax=axes[1], data=df_tsne, s=50, alpha=0.7)
    axes[1].set_title('t-SNE')
    axes[1].grid(True)

    df_umap = pd.DataFrame({
        'component_1': umap_results[:, 0],
        'component_2': umap_results[:, 1],
        'label': all_labels
    })
    sns.scatterplot(x='component_1', y='component_2', hue='label', palette='viridis', ax=axes[2], data=df_umap, s=50, alpha=0.7)
    axes[2].set_title('UMAP')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(args.output_plot_dir, f'latent_space_2_classes_visualization_count_{args.file_count_per_class}.png') # 파일명 변경
    os.makedirs(args.output_plot_dir, exist_ok=True) 
    plt.savefig(plot_path)
    print(f"✅ Latent space visualization plot saved to: {plot_path}")