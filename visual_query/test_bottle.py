import torch
import cv2
import numpy as np
from PIL import Image
import yaml
from REN import REN
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleBottleTracker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.ren = REN(self.config)
        
        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((self.config['parameters']['image_resolution'], 
                     self.config['parameters']['image_resolution'])), 
            T.ToTensor()
        ])
        
    def extract_query_features(self, query_image_path):
        """Extract features from the query image (your bottle photo)"""
        query_img = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transforms(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = self.ren(query_tensor)
        
        return query_features[0]  # Shape: [num_regions, feature_dim]
    
    def extract_frame_features(self, frame):
        """Extract features from a video frame"""
        # Convert OpenCV frame (BGR) to PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.transforms(frame_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            frame_features = self.ren(frame_tensor)
        
        return frame_features[0]  # Shape: [num_regions, feature_dim]
    
    def find_best_match(self, query_features, frame_features, top_k=5):
        """Find the most similar regions between query and frame"""
        # Normalize features for cosine similarity
        query_norm = F.normalize(query_features, p=2, dim=1)
        frame_norm = F.normalize(frame_features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(query_norm, frame_norm.T)  # [query_regions, frame_regions]
        
        # Get best matches
        max_similarities, best_matches = torch.max(similarity, dim=0)  # Best query match for each frame region
        
        # Get top-k most similar regions
        top_similarities, top_indices = torch.topk(max_similarities, min(top_k, len(max_similarities)))
        
        return top_similarities, top_indices
    
    def track_bottle(self, query_image_path, video_path, output_path=None):
        """Main tracking function"""
        print("Extracting query features...")
        query_features = self.extract_query_features(query_image_path)
        print(f"Query features shape: {query_features.shape}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
        
        # Setup video writer if output path provided
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
            similarities, region_indices = self.find_best_match(query_features, frame_features, top_k=3)
            
            # Convert region indices to approximate bounding boxes
            # This is a simplified mapping - you'd need the actual region coordinates from REN
            best_similarity = similarities[0].item()
            best_region_idx = region_indices[0].item()
            
            # Store tracking result
            tracking_results.append({
                'frame': frame_idx,
                'best_similarity': best_similarity,
                'best_region': best_region_idx
            })
            
            # Draw visualization (simplified - drawing a circle at estimated location)
            if best_similarity > 0.5:  # Threshold for detection
                # Rough estimation of region center (you'd need actual region coordinates)
                h, w = frame.shape[:2]
                regions_per_row = int(np.sqrt(frame_features.shape[0]))  # Assuming square grid
                row = best_region_idx // regions_per_row
                col = best_region_idx % regions_per_row
                
                center_y = int((row + 0.5) * h / regions_per_row)
                center_x = int((col + 0.5) * w / regions_per_row)
                
                # Draw detection
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 3)
                cv2.putText(frame, f'Sim: {best_similarity:.3f}', 
                           (center_x - 50, center_y - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
    # Initialize tracker
    tracker = SimpleBottleTracker('configs/ren_dinov2_vitl14.yaml')
    
    # Track your bottle
    results = tracker.track_bottle(
        query_image_path='query_bottle.jpg',  # Your bottle image
        video_path='test_video.mp4',          # Your bottle video
        output_path='tracked_bottle.mp4'      # Output with tracking visualization
    )
    
    # Print results
    print(f"\nTracking completed! Processed {len(results)} frames")
    avg_similarity = np.mean([r['best_similarity'] for r in results])
    print(f"Average similarity score: {avg_similarity:.3f}")
    
    # Find frames with high confidence detections
    high_conf_frames = [r for r in results if r['best_similarity'] > 0.7]
    print(f"High confidence detections: {len(high_conf_frames)} frames")

if __name__ == "__main__":
    main()