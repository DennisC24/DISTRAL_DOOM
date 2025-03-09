import gymnasium as gym
from vizdoom import DoomGame
from gymnasium.spaces import Box, Discrete
import collections
import numpy as np
import cv2

############################################################
#Defining the VizDoomGym environment
#NOTE: the action space is determined outside of the environment
# as no information is given about the action space in the config file
############################################################

class VizDoomGym(gym.Env): 
    def __init__(self, render=False, config='./VIZDOOM/scenarios/defend_the_line.cfg', max_frames=1000, width=160, height=100): 
        super(VizDoomGym, self).__init__()
        
        # Setup the game 
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(render)
        self.game.init()
        
        # Store dimensions
        self.width = width
        self.height = height
        
        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        
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
    
    ############################################################
    #We introduce a grayscale function to improve the efficiency
    # and performance of the model.
    ############################################################
    
    def grayscale(self, observation):
        width = self.width
        height = self.height
        """Convert observation to grayscale and resize."""
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resized, (height, width, 1))
        return state
    
    def close(self): 
        self.game.close()

    def save_video(self, filename="episode_video.mp4", fps=30):
        """Saves the recorded frames as a video file."""
        if not self.frames:
            print("No frames to save!")
            return
        
        # Get dimensions from first frame
        frame = self.frames[0]
        if len(frame.shape) == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze()  # Remove single channel dimension if grayscale
        height, width = frame.shape
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)
        
        try:
            for frame in self.frames:
                # Ensure frame is 2D (height, width) for grayscale
                if len(frame.shape) == 3 and frame.shape[-1] == 1:
                    frame = frame.squeeze()
                
                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Write frame
                out.write(frame)
                
        finally:
            out.release()
            print(f"Video saved as {filename}")
            # Clear frames to free memory
            self.frames.clear()

def create_vizdoom_envs(config_paths, render=False):
    """Create a list of VizDoomGym environments from config paths."""
    return [VizDoomGym(render=render, config=config_path, width=160, height=100) for config_path in config_paths]