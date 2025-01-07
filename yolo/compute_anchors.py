import mlx.core as mx
from data.voc import VOCDataset
from pathlib import Path


def compute_anchor_boxes(dataset, num_anchors=5, num_iterations=100):
    """
    Compute anchor boxes using k-means clustering with IOU distance metric
    """
    # Convert dataset to mx.array
    boxes = mx.array(dataset)
    num_boxes = len(boxes)

    # Randomly initialize clusters
    indices = mx.random.randint(0, num_boxes, (num_anchors,))
    clusters = boxes[indices]

    def iou_distance(box, clusters):
        """Calculate IOU-based distance between a box and clusters"""
        # Reshape box to [1, 2] and broadcast against clusters [num_anchors, 2]
        box = mx.expand_dims(box, axis=0)

        # Calculate intersection areas
        intersect_w = mx.minimum(box[:, 0], clusters[:, 0])
        intersect_h = mx.minimum(box[:, 1], clusters[:, 1])
        intersection = intersect_w * intersect_h

        # Calculate union areas
        box_area = box[:, 0] * box[:, 1]
        cluster_area = clusters[:, 0] * clusters[:, 1]
        union = box_area + cluster_area - intersection

        # Return 1-IOU as distance
        return 1 - intersection / (union + 1e-6)

    # K-means clustering
    for _ in range(num_iterations):
        # Calculate distances from each box to each cluster
        distances = mx.stack([iou_distance(box, clusters) for box in boxes])
        mx.eval(distances)  # Materialize distances

        # Assign boxes to nearest cluster
        assignments = mx.argmin(distances, axis=1)
        mx.eval(assignments)  # Materialize assignments

        # Store old clusters for convergence check
        old_clusters = mx.array(clusters)

        # Update clusters
        new_clusters = []
        for i in range(num_anchors):
            # Calculate mean of boxes assigned to this cluster
            # Instead of boolean indexing, use weighted sum
            weights = mx.array(assignments == i, dtype=mx.float32)
            weights = mx.expand_dims(weights, axis=1)

            # Weighted sum of boxes
            sum_boxes = mx.sum(boxes * weights, axis=0)
            count = mx.sum(weights)

            # If cluster has points, update it
            if count > 0:
                new_cluster = sum_boxes / count
            else:
                new_cluster = clusters[i]

            new_clusters.append(new_cluster)

        clusters = mx.stack(new_clusters)
        mx.eval(clusters)  # Materialize new clusters

        # Check for convergence
        if mx.array_equal(old_clusters, clusters):
            break

    # Sort anchors by area
    areas = clusters[:, 0] * clusters[:, 1]
    sorted_idx = mx.argsort(areas)
    return clusters[sorted_idx]


def update_anchors(model, dataset):
    """
    Update model's anchor boxes using the training dataset

    Args:
        model: YOLO model instance
        dataset: List of normalized bounding box dimensions [(w1,h1), (w2,h2), ...]
    """
    anchors = compute_anchor_boxes(dataset, num_anchors=model.B)
    model.anchors = anchors
    return model


def collect_box_dimensions(dataset):
    """Collect all normalized box dimensions from dataset"""
    box_dims = []

    for idx in range(len(dataset)):
        # Get image and target
        _, target = dataset[idx]

        # Target shape is [S, S, 5] where last dim is [x, y, w, h, class]
        # Extract boxes that have objects (where class > 0)
        S = target.shape[0]
        for i in range(S):
            for j in range(S):
                if target[i, j, 4] > 0:  # If there's an object
                    w, h = target[i, j, 2:4].tolist()  # Get width and height
                    box_dims.append([w, h])

    return box_dims


def main():
    # Initialize dataset
    data_dir = Path("./VOCdevkit/VOC2012")  # Adjust path as needed
    dataset = VOCDataset(
        data_dir=data_dir, year="2012", image_set="train", augment=False
    )

    print("Collecting box dimensions from dataset...")
    box_dims = collect_box_dimensions(dataset)
    print(f"Found {len(box_dims)} boxes")

    print("\nComputing anchor boxes...")
    anchors = compute_anchor_boxes(box_dims, num_anchors=5, num_iterations=100)

    print("\nAnchor boxes (width, height):")
    for i, (w, h) in enumerate(anchors.tolist()):
        print(f"Anchor {i+1}: ({w:.3f}, {h:.3f})")

    # Save anchors to file
    mx.save("anchor_boxes.npz", {"anchors": anchors})
    print("\nSaved anchor boxes to anchor_boxes.npz")


if __name__ == "__main__":
    main()
