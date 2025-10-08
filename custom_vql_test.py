import torch
import cv2
import numpy as np
from PIL import Image
import yaml
from ren import REN
import torchvision.transforms as T 
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleBottleTracker:
    def __init__(self, config_path):
        # Load REN model from config.yaml file
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.ren = REN(self.config).to(device)
        self.ren.eval()

        self.grid_size = self.config['parameters']['grid_size']

        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((self.config['parameters']['image_resolution'],
                     self.config['parameters']['image_resolution'])), 
            T.ToTensor()
        ])
        
        print(f"Initialized tracker with grid_size: {self.grid_size}")
        
    def extract_query_features(self, query_image_path):
        """Extract and average query features into single bottle representation"""
        query_img = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transforms(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = self.ren(query_tensor)
        
        # Average all regions into single bottle feature
        bottle_features = query_features[0]  # [num_regions, feature_dim]
        bottle_feature = torch.mean(bottle_features, dim=0, keepdim=True)  # [1, feature_dim]
        
        print(f"Query features averaged to shape: {bottle_feature.shape}")
        return bottle_feature

    def extract_frame_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.transforms(frame_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            frame_features = self.ren(frame_tensor)
        
        return frame_features[0]  # [num_regions, feature_dim]
    
    def find_best_match(self, query_features, frame_features, top_k=10):
        """Find the most similar regions between query and frame"""
        # Normalize features for cosine similarity
        query_norm = F.normalize(query_features, p=2, dim=1)
        frame_norm = F.normalize(frame_features, p=2, dim=1)
        
        # Similarity matrix
        similarity = torch.mm(query_norm, frame_norm.T)  # [1, frame_regions]
        
        # Get similarities for all frame regions
        similarities = similarity[0]  # [frame_regions]
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        return top_similarities, top_indices
    
    def cluster_regions(self, region_positions, similarities, cluster_distance=100):
        """Group nearby regions using simple spatial clustering"""
        if len(region_positions) == 0:
            return []
        
        # Convert to numpy for easier manipulation
        positions = np.array(region_positions)
        sims = np.array([s.item() for s in similarities])
        
        # Simple clustering: find groups of regions within cluster_distance
        clusters = []
        used = np.zeros(len(positions), dtype=bool)
        
        for i in range(len(positions)):
            if used[i]:
                continue
            
            # Start new cluster
            cluster_positions = [positions[i]]
            cluster_sims = [sims[i]]
            used[i] = True
            
            # Find nearby regions
            for j in range(i + 1, len(positions)):
                if used[j]:
                    continue
                
                # Check distance to any point in current cluster
                for cluster_pos in cluster_positions:
                    dist = np.sqrt((positions[j][0] - cluster_pos[0])**2 + 
                                 (positions[j][1] - cluster_pos[1])**2)
                    
                    if dist < cluster_distance:
                        cluster_positions.append(positions[j])
                        cluster_sims.append(sims[j])
                        used[j] = True
                        break
            
            # Calculate weighted centroid for this cluster
            cluster_sims = np.array(cluster_sims)
            cluster_positions = np.array(cluster_positions)
            
            total_weight = np.sum(cluster_sims)
            weighted_x = np.sum(cluster_positions[:, 0] * cluster_sims) / total_weight
            weighted_y = np.sum(cluster_positions[:, 1] * cluster_sims) / total_weight
            avg_sim = np.mean(cluster_sims)
            
            clusters.append({
                'center': (int(weighted_x), int(weighted_y)),
                'similarity': avg_sim,
                'size': len(cluster_positions)
            })
        
        # Sort by similarity and return best cluster
        clusters.sort(key=lambda x: x['similarity'], reverse=True)
        return clusters
    
    def track_bottle(self, query_image_path, video_path, output_path=None, 
                     similarity_threshold=0.3, top_k=10, cluster_distance=150):
        print("Extracting query features...")
        query_features = self.extract_query_features(query_image_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
        print(f"Threshold: {similarity_threshold}, top_k: {top_k}, cluster_distance: {cluster_distance}px")
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_idx = 0
        tracking_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"Processing frame {frame_idx}/{total_frames}", end='\r')
            
            # Extract features from current frame
            frame_features = self.extract_frame_features(frame)
            
            # Find top K matches
            similarities, region_indices = self.find_best_match(
                query_features, frame_features, top_k=top_k
            )
            
            # Filter by threshold
            valid_mask = similarities > similarity_threshold
            valid_similarities = similarities[valid_mask]
            valid_indices = region_indices[valid_mask]
            
            center_x, center_y, avg_similarity = None, None, 0.0
            
            if len(valid_indices) > 0:
                h, w = frame.shape[:2]
                regions_per_row = self.grid_size
                
                # Convert region indices to pixel positions
                region_positions = []
                for idx in valid_indices:
                    row = idx.item() // regions_per_row
                    col = idx.item() % regions_per_row
                    cy = int((row + 0.5) * h / regions_per_row)
                    cx = int((col + 0.5) * w / regions_per_row)
                    region_positions.append((cx, cy))
                
                # Cluster nearby regions
                clusters = self.cluster_regions(region_positions, valid_similarities, 
                                               cluster_distance=cluster_distance)
                
                if clusters:
                    # Use the best cluster (highest average similarity)
                    best_cluster = clusters[0]
                    center_x, center_y = best_cluster['center']
                    avg_similarity = best_cluster['similarity']
                    
                    # Draw main detection (best cluster center)
                    cv2.circle(frame, (center_x, center_y), 40, (0, 255, 0), 3)
                    cv2.putText(frame, f'Sim: {avg_similarity:.3f}', 
                               (center_x - 60, center_y - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Regions: {best_cluster["size"]}', 
                               (center_x - 60, center_y - 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw small dots for individual detections in the cluster
                    for cx, cy in region_positions[:best_cluster['size']]:
                        cv2.circle(frame, (cx, cy), 5, (255, 255, 0), 2)
            
            # Store results
            tracking_results.append({
                'frame': frame_idx,
                'center_x': center_x,
                'center_y': center_y,
                'avg_similarity': avg_similarity
            })
            
            # Write frame
            if output_path:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        if output_path:
            out.release()
            print(f"\nTracking video saved to: {output_path}")
        
        return tracking_results

def main():
    tracker = SimpleBottleTracker('configs/ren_dino_vitb8.yaml')
    
    # Track bottle with spatial clustering
    results = tracker.track_bottle(
        query_image_path='query_bottle.jpg',
        video_path='test_video.mp4',
        output_path='tracked_bottle.mp4',
        similarity_threshold=0.3,  # Adjust if needed
        top_k=10,  # Look at more regions
        cluster_distance=150  # Pixels - group regions within this distance
    )
    
    # Analysis
    print(f"\nTracking completed. Processed {len(results)} frames")
    
    detected_frames = [r for r in results if r['center_x'] is not None]
    print(f"Frames with detection: {len(detected_frames)}/{len(results)} ({100*len(detected_frames)/len(results):.1f}%)")
    
    if detected_frames:
        similarities = [r['avg_similarity'] for r in detected_frames]
        print(f"Similarity - Avg: {np.mean(similarities):.3f}, Max: {np.max(similarities):.3f}, Min: {np.min(similarities):.3f}")

if __name__ == "__main__":
    main()