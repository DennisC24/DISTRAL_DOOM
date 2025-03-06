import gymnasium as gym
from vizdoom import DoomGame
from gymnasium.spaces import Box, Discrete
import collections
import numpy as np
import cv2


class VizDoomGym(gym.Env): 
    def __init__(self, render=False, config='./VIZDOOM/scenarios/defend_the_line.cfg', max_frames=1000): 
        super(VizDoomGym, self).__init__()
        
        # Setup the game 
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(render)
        self.game.init()
        
        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        
        # Store frames for video recording
        self.frames = collections.deque(maxlen=max_frames)

    def step(self, action):
        # Create action array based on action dimension
        actions = np.identity(self.action_dim)
        movement_reward = self.game.make_action(actions[action], 4)
        
        # Set the reward
        reward = movement_reward
        
        # Get the state, terminated, truncated, and info
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            self.frames.append(state)  # Capture each frame
            info = {"game_variables": self.game.get_state().game_variables}
        else: 
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {"game_variables": 0}
        
        # Check if episode is finished
        terminated = self.game.is_episode_finished()
        truncated = False  # Assuming truncation is not applied here
        
        return state, reward, terminated, truncated, info
    
    def render(self, mode="human"): 
        pass
    
    def reset(self, seed=None, options=None):
        """Reset the environment with an optional seed"""
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)  # Set VizDoom's internal seed
        
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state), {}  # Return state and empty info dict
    
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resized, (100, 160, 1))
        return state
    
    def close(self): 
        self.game.close()

    def save_video(self, filename="episode_video.mp4", fps=30):
        """Saves the recorded frames as a video file."""
        if not self.frames:
            print("No frames to save!")
            return
        
        # Set up video writer
        height, width = self.frames[0].shape[:2]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
        
        for frame in self.frames:
            # The VideoWriter expects frames in color, so we replicate the grayscale channel
            color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(color_frame)
        
        out.release()
        print(f"Video saved as {filename}")

def create_vizdoom_envs(config_paths, render=False):
    """Create a list of VizDoomGym environments from config paths."""
    return [VizDoomGym(render=render, config=config_path) for config_path in config_paths]