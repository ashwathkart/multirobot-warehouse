import json
import os
from datetime import datetime
from typing import Dict, Any, List


class ResultsLogger:
    """
    Logs simulation results with timestamps.
    Saves GIFs and statistics to timestamped files.
    """

    def __init__(self, output_dir: str = 'logs'):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.stats: Dict[str, Any] = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def get_filename(self, extension: str) -> str:
        """Get timestamped filename"""
        return os.path.join(self.output_dir, f'{self.timestamp}.{extension}')

    def set_stat(self, key: str, value: Any):
        """Set a statistic"""
        self.stats[key] = value

    def set_stats(self, stats_dict: Dict[str, Any]):
        """Set multiple statistics at once"""
        self.stats.update(stats_dict)

    def save_gif(self, frames: List, fps: int) -> str:
        """
        Save frames as GIF.

        Args:
            frames: List of frames
            fps: Frames per second

        Returns:
            Path to saved GIF file
        """
        import imageio

        if not frames:
            print("No frames to save")
            return None

        gif_path = self.get_filename('gif')
        frames_fixed = [frame.swapaxes(0, 1) for frame in frames]
        imageio.mimsave(gif_path, frames_fixed, fps=fps)
        print(f"Saved GIF to {gif_path}")
        return gif_path

    def save_stats(self) -> str:
        """
        Save statistics as JSON.

        Returns:
            Path to saved stats file
        """
        # Add timestamp to stats
        self.stats['timestamp'] = self.timestamp
        self.stats['datetime'] = datetime.now().isoformat()

        stats_path = self.get_filename('json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"Saved stats to {stats_path}")
        return stats_path

    def save_all(self, frames: List = None, fps: int = 2) -> Dict[str, str]:
        """
        Save all results (GIF and stats).

        Args:
            frames: List of frames for GIF
            fps: Frames per second

        Returns:
            Dictionary with paths to saved files
        """
        results = {}

        if frames:
            gif_path = self.save_gif(frames, fps)
            if gif_path:
                results['gif'] = gif_path

        stats_path = self.save_stats()
        if stats_path:
            results['stats'] = stats_path

        return results

    def print_summary(self):
        """Print simulation summary"""
        print(f"\nSimulation Results Summary ({self.timestamp}):")
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
