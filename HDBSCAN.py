import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================
# Setting
# ================================
FILENAME = '1000vect2(1.00,0.30,2).txt'  
BOX_SIZE = np.array([1000.0, 1000.0, 1000.0])
MIN_CLUSTER_SIZE = 10
POS_WEIGHT = 0.5

# ================================
# Distance matrix computation function (PBC + normalize + equal weight)
# ================================
def compute_custom_distance(positions, directions, box_size=BOX_SIZE):
    N = len(positions)
    dist_matrix = np.zeros((N, N))
    max_spatial = np.linalg.norm(box_size / 2.0)
    for i in range(N):
        for j in range(i+1, N):
            delta = np.abs(positions[i] - positions[j])
            delta = np.minimum(delta, box_size - delta)
            d_pos = np.linalg.norm(delta)
            cos_sim = np.dot(directions[i], directions[j])
            d_dir = 1.0 - cos_sim
            d_pos_norm = d_pos / max_spatial
            d_dir_norm = d_dir / 2.0
            dist = POS_WEIGHT * d_pos_norm + (1-POS_WEIGHT) * d_dir_norm
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

# ================================
# main part
# ================================
def main():
    try:
        raw_text = open(FILENAME, 'r', encoding='utf-8').read().strip()
    except Exception as e:
        print(f"failed: {e}")
        return

    blocks = raw_text.split('\n\n')
    results = []

    for frame_idx, blk in enumerate(blocks, start=1):
        # read data
        try:
            data = np.loadtxt(blk.splitlines(), delimiter=',')
        except Exception as e:
            print(f"[Frame {frame_idx}] failed1: {e}")
            continue
        if data.ndim != 2 or data.shape[1] != 6:
            print(f"[Frame {frame_idx}] failed2")
            continue

        positions  = data[:, :3]
        directions = data[:, 3:6]
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(norms==0, 1, norms)

        # Distance matrix → clustering
        dist_mat = compute_custom_distance(positions, directions)
        labels = hdbscan.HDBSCAN(metric='precomputed',
                                 min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(dist_mat)

        
        N_total = len(labels)
        N_clustered = np.sum(labels >= 0)
        pct_clustered = 100.0 * N_clustered / N_total

        # get clusters
        valid = labels >= 0
        unique_clusters = np.unique(labels[valid])

        # calcurate order parameter for each cluster
        order_params = []
        for c in unique_clusters:
            mask = (labels == c)
            v_sum = directions[mask].sum(axis=0)
            phi = np.linalg.norm(v_sum) / mask.sum()
            order_params.append(phi)
        if not order_params:
            order_params = [0.0]

        results.append({
            'order_params': order_params,
            'pct_clustered': pct_clustered
        })
        print(f"[Frame {frame_idx}] clusters={len(order_params)}, "
              f"order_params={order_params}, "
              f"%clustered={pct_clustered:.1f}%")

        # visualize
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        # coloring each cluster
        for c in unique_clusters:
            mask = labels == c
            ax.scatter(
                positions[mask,0], positions[mask,1], positions[mask,2],
                label=f"Cluster {c}", s=20
            )
            vel_mag = np.linalg.norm(directions[mask], axis=1)
            scaled = directions[mask] * vel_mag[:,None] * 50.0
            ax.quiver(
                positions[mask,0], positions[mask,1], positions[mask,2],
                scaled[:,0], scaled[:,1], scaled[:,2],
                normalize=False, arrow_length_ratio=0.2, linewidth=0.5
            )

        # disp noise
        noise_mask = (labels == -1)
        if noise_mask.any():
            ax.scatter(
                positions[noise_mask,0], positions[noise_mask,1], positions[noise_mask,2],
                c='k', marker='x', label='Noise', s=20
            )

        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()

    # ================================
    # Export results to a text file
    # ================================
    out_fname = 'order_params_summary.txt'
    with open(out_fname, 'w') as f:
        for r in results:
            line = ' '.join(f"{phi:.6f}" for phi in r['order_params'])
           
            line += f"  {r['pct_clustered']:.2f}%"
            f.write(line + '\n')
    print(f"Order parameters summary saved to '{out_fname}'")

if __name__ == '__main__':
    main()
