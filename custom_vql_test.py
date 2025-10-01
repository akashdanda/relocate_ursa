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
        # Load config and REN model
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.ren = REN(self.config)
        
        # Get REN's actual grid points
        self.grid_points = self.ren.grid_points
        self.grid_size = self.ren.grid_size
        self.image_resolution = self.config['parameters']['image_resolution']
        
        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((self.image_resolution, self.image_resolution)),
            T.ToTensor()
        ])
        
    def extract_query_features(self, query_image_path):
        """Extract features for all grid regions in query image"""
        query_img = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transforms(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = self.ren(query_tensor)
        
        return query_features[0]  # [num_regions, feature_dim]

    def extract_frame_features(self, frame):
        """Extract features for all grid regions in frame"""
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
        
        # Get best matches
        max_similarities, best_matches = torch.max(similarity, dim=0)
        
        # Get top k most similar regions
        top_similarities, top_indices = torch.topk(
            max_similarities, 
            min(top_k, len(max_similarities))
        )
        
        return top_similarities, top_indices
    
    def get_grid_coordinates(self, region_idx, frame_shape):
        """
        Convert region index to pixel coordinates using REN's actual grid.
        
        Args:
            region_idx: Index of the region in the grid
            frame_shape: (height, width) of the actual video frame
        
        Returns:
            (center_x, center_y): Pixel coordinates in the frame
        """
        # Get the grid point (y, x) coordinates from REN's grid
        grid_point = self.grid_points[region_idx]
        y_grid = grid_point[0].item()
        x_grid = grid_point[1].item()
        
        # Scale from image_resolution to actual frame size
        scale_y = frame_shape[0] / self.image_resolution
        scale_x = frame_shape[1] / self.image_resolution
        
        center_y = int(y_grid * scale_y)
        center_x = int(x_grid * scale_x)
        
        return (center_x, center_y)
    
    def track_bottle(self, query_image_path, video_path, output_path=None):
        """
        Track bottle in video using REN's grid-based approach.
        
        Args:
            query_image_path: Path to image with the bottle
            video_path: Path to video file
            output_path: Optional path to save output video
        """
        print("Extracting query features...")
        query_features = self.extract_query_features(query_image_path)
        print(f"Query features shape: {query_features.shape}")
        print(f"Grid size: {self.grid_size}x{self.grid_size} = {len(self.grid_points)} points")
        
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
            similarities, region_indices = self.find_best_match(
                query_features, frame_features, top_k=3
            )
            
            # Get best match
            best_similarity = similarities[0].item()
            best_region_idx = region_indices[0].item()
            
            # Store tracking result
            tracking_results.append({
                'frame': frame_idx,
                'best_similarity': best_similarity,
                'best_region': best_region_idx
            })
            
            # Draw visualization if confidence is high
            if best_similarity > 0.5:
                # Get actual grid coordinates
                center_x, center_y = self.get_grid_coordinates(
                    best_region_idx, 
                    (frame_height, frame_width)
                )
                
                # Draw detection
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 3)
                cv2.putText(
                    frame, 
                    f'Sim: {best_similarity:.3f}', 
                    (center_x - 50, center_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
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
    
    # Track the bottle
    results = tracker.track_bottle(
        query_image_path='query_bottle.jpg',
        video_path='test_video.mp4',
        output_path='tracked_bottle.mp4'
    )
    
    # Print results
    print(f"\nTracking completed. Processed {len(results)} frames")
    
    # Calculate statistics
    avg_similarity = np.mean([r['best_similarity'] for r in results])
    print(f"Average similarity score: {avg_similarity:.3f}")
    
    # Find frames with high confidence detections
    high_conf_frames = [r for r in results if r['best_similarity'] > 0.7]
    print(f"High confidence detections (>0.7): {len(high_conf_frames)} frames")
    
    medium_conf_frames = [r for r in results if 0.5 < r['best_similarity'] <= 0.7]
    print(f"Medium confidence detections (0.5-0.7): {len(medium_conf_frames)} frames")
    
    low_conf_frames = [r for r in results if r['best_similarity'] <= 0.5]
    print(f"Low confidence detections (<=0.5): {len(low_conf_frames)} frames")


if __name__ == "__main__":
    main()