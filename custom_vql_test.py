import torch
import cv2
import numpy as np
from PIL import Image
import yaml
from ren import REN
import torchvision.transforms as T 
import torch.nn.functional as F
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleBottleTracker:
    def __init__(self, config_path):
        # Load REN model from config.yaml file
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.ren = REN(self.config).to(device)
        
        # Get checkpoint path from config
        self.ren_checkpoint = self.config['parameters']['ren_ckpt']
        print(f"Checkpoint path: {self.ren_checkpoint}")
        
        # CRITICAL: Load the trained weights
        self.load_ren()
        
        self.ren.eval()
        
        # Use grid_size from config
        self.grid_size = self.config['parameters']['grid_size']
        
        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((self.config['parameters']['image_resolution'],
                     self.config['parameters']['image_resolution'])), 
            T.ToTensor()
        ])
        
        print(f"✓ Initialized tracker with grid_size: {self.grid_size}")
    
    def load_ren(self):
        """Load REN checkpoint with trained weights"""
        if os.path.exists(self.ren_checkpoint):
            print(f"Loading REN checkpoint from: {self.ren_checkpoint}")
            checkpoint = torch.load(self.ren_checkpoint, map_location=device)
            
            # Load the region encoder weights
            self.ren.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            
            ren_epoch = checkpoint.get('epoch', 'unknown')
            ren_iter = checkpoint.get('iter_count', 'unknown')
            print(f'✓ Loaded REN checkpoint: {ren_epoch} epochs, {ren_iter} iterations.')
        else:
            print(f'ERROR: No REN checkpoint found at: {self.ren_checkpoint}')
            print(f'Full path: {os.path.abspath(self.ren_checkpoint)}')
            print('The model will use random weights, which will NOT work properly!')
            print('Please download or train the checkpoint first.')
            exit()
        
    def extract_query_features(self, query_image_path):
        """Extract features from query image"""
        query_img = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transforms(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = self.ren(query_tensor)
        
        return query_features[0]  # [num_regions, feature_dim]

    def extract_frame_features(self, frame):
        """Extract features from video frame"""
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
                     similarity_threshold=0.3, top_k=5, use_weighted_centroid=True):
        print("\nExtracting query features...")
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
        print(f"Threshold: {similarity_threshold}, top_k: {top_k}, weighted_centroid: {use_weighted_centroid}\n")
        
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
            
            # Find best matches
            similarities, region_indices = self.find_best_match(query_features, frame_features, top_k=top_k)
            
            # Filter by threshold
            valid_mask = similarities > similarity_threshold
            valid_similarities = similarities[valid_mask]
            valid_indices = region_indices[valid_mask]
            
            center_x, center_y, avg_similarity = None, None, 0.0
            
            if len(valid_indices) > 0:
                h, w = frame.shape[:2]
                regions_per_row = self.grid_size  # Use actual grid_size from config
                
                if use_weighted_centroid and len(valid_indices) > 1:
                    # Weighted centroid approach - groups nearby detections
                    total_weight = 0
                    weighted_x = 0
                    weighted_y = 0
                    
                    for sim, idx in zip(valid_similarities, valid_indices):
                        row = idx.item() // regions_per_row
                        col = idx.item() % regions_per_row
                        cy = (row + 0.5) * h / regions_per_row
                        cx = (col + 0.5) * w / regions_per_row
                        
                        weight = sim.item()
                        weighted_x += cx * weight
                        weighted_y += cy * weight
                        total_weight += weight
                    
                    center_x = int(weighted_x / total_weight)
                    center_y = int(weighted_y / total_weight)
                    avg_similarity = total_weight / len(valid_similarities)
                else:
                    # Just use best match
                    best_idx = valid_indices[0].item()
                    row = best_idx // regions_per_row
                    col = best_idx % regions_per_row
                    center_y = int((row + 0.5) * h / regions_per_row)
                    center_x = int((col + 0.5) * w / regions_per_row)
                    avg_similarity = valid_similarities[0].item()
                
                # Draw detection
                cv2.circle(frame, (center_x, center_y), 35, (0, 255, 0), 3)
                cv2.putText(frame, f'Sim: {avg_similarity:.3f}', 
                           (center_x - 60, center_y - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Optional: show individual matching regions as small dots
                if use_weighted_centroid and len(valid_indices) > 1:
                    for idx in valid_indices:
                        row = idx.item() // regions_per_row
                        col = idx.item() % regions_per_row
                        cy = int((row + 0.5) * h / regions_per_row)
                        cx = int((col + 0.5) * w / regions_per_row)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), 2)  # Small yellow dots
            
            # Store tracking result
            tracking_results.append({
                'frame': frame_idx,
                'center_x': center_x,
                'center_y': center_y,
                'avg_similarity': avg_similarity,
                'num_regions': len(valid_indices)
            })
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        if output_path:
            out.release()
            print(f"\n✓ Tracking video saved to: {output_path}")
        
        return tracking_results

def main():
    tracker = SimpleBottleTracker('configs/ren_dino_vitb8.yaml')
    
    # Track bottle
    results = tracker.track_bottle(
        query_image_path='query_bottle.jpg',
        video_path='test_video.mp4',
        output_path='tracked_bottle.mp4',
        similarity_threshold=0.5,  # Adjust based on your results
        top_k=5,
        use_weighted_centroid=True  # Set False to use only best match
    )
    
    # Analysis
    print(f"\n{'='*50}")
    print(f"Tracking Results:")
    print(f"{'='*50}")
    print(f"Total frames processed: {len(results)}")
    
    detected_frames = [r for r in results if r['center_x'] is not None]
    print(f"Frames with detection: {len(detected_frames)}/{len(results)} ({100*len(detected_frames)/len(results):.1f}%)")
    
    if detected_frames:
        similarities = [r['avg_similarity'] for r in detected_frames]
        print(f"Similarity - Avg: {np.mean(similarities):.3f}, Max: {np.max(similarities):.3f}, Min: {np.min(similarities):.3f}")
        
        # Show detection confidence distribution
        for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
            count = len([r for r in detected_frames if r['avg_similarity'] > threshold])
            print(f"Detections > {threshold}: {count} frames ({100*count/len(results):.1f}%)")

if __name__ == "__main__":
    main()