import torch
import cv2
import numpy as np
from PIL import Image
import yaml
from ren import REN
import torchvision.transforms as T 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleBottleTracker:
    def __init__(self, config_path):
        #load ren model from config.yaml file
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.ren = REN(self.config)
        
        # Image preprocessing Compose multiple transformations at once
        self.transforms = T.Compose([
            T.Resize((self.config['parameters']['image_resolution'], #resize all images to same size based on config 
                     self.config['parameters']['image_resolution'])), 
            T.ToTensor() #converts PIL image to PyTorch tensor(pixel vals normalized between 0 & 1)
        ])
        
        # Cache for grid dimensions - will adapt to video aspect ratio
        self.grid_h = None
        self.grid_w = None
        self.use_adaptive_grid = self.config['parameters'].get('use_adaptive_grid', True)  # Enable rectangular grids
        
    def _calculate_grid_dimensions(self, num_regions, image_height, image_width):
        """
        Calculate actual grid dimensions based on number of regions and image aspect ratio.
        Supports rectangular grids that match the video's aspect ratio.
        """
        if self.grid_h is not None and self.grid_w is not None:
            return self.grid_h, self.grid_w
        
        # Check if using SLIC (non-grid based)
        if self.config['parameters'].get('use_slic', False):
            # For SLIC, still use square grid as approximation for visualization
            self.grid_h = int(np.sqrt(num_regions))
            self.grid_w = int(np.sqrt(num_regions))
            print(f"Using SLIC with approximate grid: {self.grid_h}x{self.grid_w}")
            return self.grid_h, self.grid_w
        
        # Method 1: If adaptive grid is disabled, use square grid
        if not self.use_adaptive_grid:
            self.grid_h = int(np.sqrt(num_regions))
            self.grid_w = int(np.sqrt(num_regions))
            print(f"Using square grid: {self.grid_h}x{self.grid_w}")
            return self.grid_h, self.grid_w
        
        # Method 2: Calculate rectangular grid based on aspect ratio
        aspect_ratio = image_width / image_height
        
        # Find factors of num_regions that best match aspect ratio
        best_diff = float('inf')
        best_h, best_w = int(np.sqrt(num_regions)), int(np.sqrt(num_regions))
        
        # Try all possible factorizations
        for h in range(1, num_regions + 1):
            if num_regions % h == 0:
                w = num_regions // h
                grid_aspect = w / h
                diff = abs(grid_aspect - aspect_ratio)
                
                # Prefer factorizations closer to the aspect ratio
                if diff < best_diff:
                    best_diff = diff
                    best_h, best_w = h, w
        
        self.grid_h = best_h
        self.grid_w = best_w
        
        grid_aspect = self.grid_w / self.grid_h
        print(f"Adaptive rectangular grid: {self.grid_h}x{self.grid_w} "
              f"(grid aspect: {grid_aspect:.2f}, video aspect: {aspect_ratio:.2f})")
        return self.grid_h, self.grid_w
        
    def extract_query_features(self, query_image_path):
        query_img = Image.open(query_image_path).convert('RGB') #convert image from brg to rgb
        query_tensor = self.transforms(query_img).unsqueeze(0).to(device) #4d tensor( [batchsize, channels, height, width]) -> [1, 3, h, w]
        
        with torch.no_grad():
            query_features = self.ren(query_tensor) #applies ren model
            # REN returns [batch_size, num_regions, feature_dim]
            # Remove batch dimension to get [num_regions, feature_dim]
            query_features = query_features[0]
        
        return query_features  # [num_regions, feature_dim]

    def extract_frame_features(self, frame):
        #same as query but for individual frames
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.transforms(frame_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            frame_features = self.ren(frame_tensor)
            # REN returns [batch_size, num_regions, feature_dim]
            # Remove batch dimension to get [num_regions, feature_dim]
            frame_features = frame_features[0]
        
        return frame_features  # [num_regions, feature_dim] 
    
    def find_best_match(self, query_features, frame_features, top_k=5):
        """Find the most similar regions between query and frame"""
        # Normalize features for cosine similarity
        query_norm = F.normalize(query_features, p=2, dim=1)
        frame_norm = F.normalize(frame_features, p=2, dim=1)
        
        #similarity matrix
        similarity = torch.mm(query_norm, frame_norm.T)  # [query_regions, frame_regions]
        
        # Get best matches(closest to 1)
        max_similarities, best_matches = torch.max(similarity, dim=0)  # Best query match for each frame region
        
        # Get top most similar regions
        top_similarities, top_indices = torch.topk(max_similarities, min(top_k, len(max_similarities)))
        
        return top_similarities, top_indices
    
    def region_idx_to_coords(self, region_idx, frame_height, frame_width, num_regions):
        """
        Convert region index to pixel coordinates.
        Now handles non-square grids properly.
        """
        grid_h, grid_w = self._calculate_grid_dimensions(num_regions, frame_height, frame_width)
        
        # Calculate row and column in the grid
        row = region_idx // grid_w
        col = region_idx % grid_w
        
        # Calculate region dimensions
        region_h = frame_height / grid_h
        region_w = frame_width / grid_w
        
        # Calculate center coordinates
        center_x = int((col + 0.5) * region_w)
        center_y = int((row + 0.5) * region_h)
        
        # Also return bounding box corners
        x1 = int(col * region_w)
        y1 = int(row * region_h)
        x2 = int((col + 1) * region_w)
        y2 = int((row + 1) * region_h)
        
        return center_x, center_y, (x1, y1, x2, y2)
    
    def track_bottle(self, query_image_path, video_path, output_path=None):
        print("Extracting query features...")
        query_features = self.extract_query_features(query_image_path) #just one time query feature extraction
        print(f"Query features shape: {query_features.shape}") #-[regions, dimensional shape]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        #get all dim & #frames
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)) #create video out with same prop as inp
        
        frame_idx = 0
        tracking_results = []
        
        while True:
            #read and actual frame
            ret, frame = cap.read()
            if not ret: #break if end
                break
                
            print(f"Processing frame {frame_idx}/{total_frames}", end='\r')
            
            # Extract features from current frame
            frame_features = self.extract_frame_features(frame)
            
            # Find best matches
            similarities, region_indices = self.find_best_match(query_features, frame_features, top_k=3)#with cosine similarity matrix, corresponding region within frame
            
            #alr sorted from torch().topk, just pick 0th index from both
            best_similarity = similarities[0].item() 
            best_region_idx = region_indices[0].item() 
            
            # Store tracking result
            tracking_results.append({
                'frame': frame_idx,
                'best_similarity': best_similarity,
                'best_region': best_region_idx
            })
            
            # drawing visual
            if best_similarity > 0.5:  #only add circle if greater than 0.5 for sim
                h, w = frame.shape[:2] #for cur video frame
                num_regions = frame_features.shape[0]
                
                # Use new function to get coordinates - handles non-square grids
                center_x, center_y, (x1, y1, x2, y2) = self.region_idx_to_coords(
                    best_region_idx, h, w, num_regions
                )
                
                # Draw detection with both circle and bounding box
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 3) #30 for radius, bgr(green), 3 thickness
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Add bounding box
                cv2.putText(frame, f'Sim: {best_similarity:.3f}', 
                           (center_x - 50, center_y - 40), #text placement left of circle center and up
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #.7 is the font sacle
            
            # adding frame to output vid
            if output_path:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release() #release vid
        if output_path:
            out.release()
            print(f"\nTracking video saved to: {output_path}")
        
        return tracking_results

def main():
    tracker = SimpleBottleTracker('configs/ren_dino_vitb8.yaml') #tracker intialization
    
    #results
    results = tracker.track_bottle(
        query_image_path='query_bottle.jpg',  #bottle image
        video_path='test_video.mp4',          #bottle video
        output_path='tracked_bottle.mp4'      #output
    )
    
    # Print results
    print(f"\nTracking completed. Processed {len(results)} frames")
    avg_similarity = np.mean([r['best_similarity'] for r in results])
    print(f"Average similarity score: {avg_similarity:.3f}")
    
    # Find frames with high confidence detections
    high_conf_frames = [r for r in results if r['best_similarity'] > 0.7]
    print(f"High confidence detections: {len(high_conf_frames)} frames")

if __name__ == "__main__":
    main()