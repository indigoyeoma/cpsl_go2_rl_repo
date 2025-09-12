    def _display_single_camera(self):
        """Display single camera feed (RGB + depth side by side)."""
        # Prepare RGB display
        rgb_display = np.zeros((400, 400, 3), dtype=np.uint8)
        if hasattr(self, 'rgb_image') and self.rgb_image is not None:
            rgb_img = self.rgb_image.cpu().numpy()
            
            if rgb_img.size > 0 and not np.all(rgb_img == 0):
                # Convert from RGBA to BGR for OpenCV display
                if rgb_img.shape[-1] == 4:  # RGBA
                    rgb_img = rgb_img[:, :, :3]  # Remove alpha channel
                
                # Convert RGB to BGR for OpenCV
                rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                rgb_display = cv2.resize(rgb_img_bgr, (400, 400))
            else:
                cv2.putText(rgb_display, "NO RGB DATA", (120, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(rgb_display, "NO RGB DATA", (120, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Prepare depth display
        depth_display = np.zeros((400, 400, 3), dtype=np.uint8)
        if hasattr(self, 'raw_depth_image') and self.raw_depth_image is not None:
            depth_img = self.raw_depth_image.cpu().numpy()
            
            # Check if depth has valid values
            valid_mask = (depth_img > self.cfg.depth.near_clip) & (depth_img < self.cfg.depth.far_clip)
            if np.any(valid_mask):
                # Normalize depth values for visualization
                depth_normalized = np.clip(depth_img, self.cfg.depth.near_clip, self.cfg.depth.far_clip)
                depth_normalized = (depth_normalized - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)
                
                # Invert so close objects are bright (white) and far objects are dark (black)
                depth_normalized = 1.0 - depth_normalized
                
                # Convert to 8-bit and apply to display
                depth_normalized = (depth_normalized * 255).astype(np.uint8)
                depth_resized = cv2.resize(depth_normalized, (400, 400))
                depth_display = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
            else:
                cv2.putText(depth_display, "NO VALID DEPTH", (90, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(depth_display, "NO DEPTH DATA", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Combine RGB and depth side by side
        combined_display = np.hstack([rgb_display, depth_display])
        
        # Add text overlays
        cv2.putText(combined_display, "RGB", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_display, "DEPTH", (410, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_display, f"Env {self.camera_env_id}/{self.num_envs}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_display, "Press 'q' to close", (10, 375), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.camera_window_name, combined_display)