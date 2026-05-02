import imageio
import pygame
from config import FPS


class FrameManager:
    """Manages frame capture and video/GIF generation"""

    def __init__(self, enable_rendering=True):
        self.enable_rendering = enable_rendering
        self.frames = []

    def capture_frame(self, surface):
        """Capture a frame from a pygame surface"""
        if self.enable_rendering:
            # Convert pygame surface to numpy array
            frame = pygame.surfarray.array3d(surface)
            self.frames.append(frame)

    def save_gif(self, filename):
        """Save captured frames as GIF"""
        if not self.frames:
            print("No frames to save")
            return

        # Fix frame orientation (swap x and y axes)
        frames = [frame.swapaxes(0, 1) for frame in self.frames]
        imageio.mimsave(filename, frames, fps=FPS)
        print(f"Saved GIF to {filename}")

    def clear(self):
        """Clear all captured frames"""
        self.frames = []

    def get_frame_count(self):
        """Get number of frames captured"""
        return len(self.frames)
