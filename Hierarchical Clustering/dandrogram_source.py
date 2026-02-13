import numpy as np

def dandrogram_source(estimator):
    counts = np.zeros(estimator.children_.shape[0])
    n_samples = len(estimator.labels_)

    for i, merge in enumerate(estimator.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [estimator.children_, estimator.distances_, counts]
    ).astype(float)

    # 시각화에 필요한 배열 리턴
    return linkage_matrix