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
        query_img = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transforms(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = self.ren(query_tensor)
        
        return query_features[0]  # [num_regions, feature_dim]

    def extract_frame_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.transforms(frame_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            frame_features = self.ren(frame_tensor)
        
        return frame_features[0] 
    
    def find_best_match(self, query_features, frame_features, top_k=5):
        """Find the most similar regions between query and frame"""
        # Normalize features for cosine similarity
        query_norm = F.normalize(query_features, p=2, dim=1)
        frame_norm = F.normalize(frame_features, p=2, dim=1)
        
        # Similarity matrix
        similarity = torch.mm(query_norm, frame_norm.T)  # [query_regions, frame_regions]
        
        # Get best matches (closest to 1)
        max_similarities, best_matches = torch.max(similarity, dim=0)
        
        # Get top most similar regions
        top_similarities, top_indices = torch.topk(max_similarities, min(top_k, len(max_similarities)))
        
        return top_similarities, top_indices
    
    def track_bottle(self, query_image_path, video_path, output_path=None, 
                     similarity_threshold=0.3, top_k=5):
        print("Extracting query features...")
        query_features = self.extract_query_features(query_image_path)
        print(f"Query features shape: {query_features.shape}")
        
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
        print(f"Using similarity threshold: {similarity_threshold}, top_k: {top_k}")
        
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
            
            if len(valid_indices) > 0:
                h, w = frame.shape[:2]
                regions_per_row = self.grid_size
                
                # Calculate weighted centroid based on similarity scores
                total_weight = 0
                weighted_x = 0
                weighted_y = 0
                
                for sim, idx in zip(valid_similarities, valid_indices):
                    row = idx.item() // regions_per_row
                    col = idx.item() % regions_per_row
                    center_y = (row + 0.5) * h / regions_per_row
                    center_x = (col + 0.5) * w / regions_per_row
                    
                    # Use similarity as weight
                    weight = sim.item()
                    weighted_x += center_x * weight
                    weighted_y += center_y * weight
                    total_weight += weight
                
                if total_weight > 0:
                    # Calculate final weighted centroid
                    center_x = int(weighted_x / total_weight)
                    center_y = int(weighted_y / total_weight)
                    avg_similarity = total_weight / len(valid_similarities)
                    
                    # Store tracking result
                    tracking_results.append({
                        'frame': frame_idx,
                        'center_x': center_x,
                        'center_y': center_y,
                        'avg_similarity': avg_similarity,
                        'num_regions': len(valid_indices)
                    })
                    
                    # Draw main detection (weighted centroid)
                    cv2.circle(frame, (center_x, center_y), 35, (0, 255, 0), 3)
                    cv2.putText(frame, f'Sim: {avg_similarity:.3f}', 
                               (center_x - 60, center_y - 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Optional: Draw small dots for individual region detections
                    for idx in valid_indices:
                        row = idx.item() // regions_per_row
                        col = idx.item() % regions_per_row
                        cy = int((row + 0.5) * h / regions_per_row)
                        cx = int((col + 0.5) * w / regions_per_row)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), 2)  # Small yellow dots
            else:
                # No detection in this frame
                tracking_results.append({
                    'frame': frame_idx,
                    'center_x': None,
                    'center_y': None,
                    'avg_similarity': 0.0,
                    'num_regions': 0
                })
            
            # Write frame to output video
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
    
    # Track with weighted centroid approach
    results = tracker.track_bottle(
        query_image_path='query_bottle.jpg',
        video_path='test_video.mp4',
        output_path='tracked_bottle.mp4',
        similarity_threshold=0.3,  # Adjust based on your needs
        top_k=5  # Consider top 5 matching regions
    )
    
    # Analysis
    print(f"\nTracking completed. Processed {len(results)} frames")
    
    # Calculate statistics
    detected_frames = [r for r in results if r['center_x'] is not None]
    print(f"Frames with detection: {len(detected_frames)}/{len(results)} ({100*len(detected_frames)/len(results):.1f}%)")
    
    if detected_frames:
        avg_similarity = np.mean([r['avg_similarity'] for r in detected_frames])
        avg_regions = np.mean([r['num_regions'] for r in detected_frames])
        print(f"Average similarity: {avg_similarity:.3f}")
        print(f"Average regions matched per frame: {avg_regions:.1f}")
        
        # High confidence detections
        for threshold in [0.4, 0.5, 0.6, 0.7]:
            high_conf = len([r for r in detected_frames if r['avg_similarity'] > threshold])
            print(f"Detections > {threshold}: {high_conf} frames ({100*high_conf/len(results):.1f}%)")

if __name__ == "__main__":
    main()