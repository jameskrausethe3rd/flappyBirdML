import pygame
import random
import numpy as np

SCREEN_WIDTH, SCREEN_HEIGHT = 288, 512
PIPE_GAP = 100
GRAVITY = 0.25
JUMP_STRENGTH = -4.5
BIRD_X = 50
BIRD_SIZE = 20
PIPE_WIDTH = 52
PIPE_DISTANCE = 200 

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.reset()

        self.pipe_img = pygame.image.load("assets/pipe.png").convert_alpha()
        self.pipe_img = pygame.transform.scale(self.pipe_img, (50, SCREEN_HEIGHT))

        self.bird_img = pygame.image.load("assets/bird.png").convert_alpha()
        self.bird_img = pygame.transform.scale(self.bird_img, (34, 24))
        self.bird_angle = 0 
        
        self.bg_img = pygame.image.load("assets/background.jpg").convert()
        self.bg_img = pygame.transform.scale(self.bg_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.frames_since_last_flap = 0
        self.score = 0

        # Create first pipe
        self.pipes = [{
            "x": SCREEN_WIDTH,
            "gap_y": random.randint(100, 300)
        }]
        return self._get_state()

    def _get_state(self):
        next_pipe = self.pipes[0]
        return np.array([
            self.bird_y / SCREEN_HEIGHT,
            self.bird_vel / 10.0,
            (next_pipe["x"] - 50) / SCREEN_WIDTH,
            (next_pipe["gap_y"] - self.bird_y) / SCREEN_HEIGHT,
            (next_pipe["gap_y"] + PIPE_GAP / 2 - self.bird_y) / SCREEN_HEIGHT,
            (next_pipe["gap_y"] - PIPE_GAP / 2 - self.bird_y) / SCREEN_HEIGHT
        ], dtype=np.float32)

    def step(self, action):
        self.frames_since_last_flap += 1
        if action == 1 and self.frames_since_last_flap > 5:
            self.bird_vel = JUMP_STRENGTH
            self.frames_since_last_flap = 0

        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        # Move all pipes
        for pipe in self.pipes:
            pipe['x'] -= 2

        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + PIPE_WIDTH > 0]

        # Add new pipe if needed
        if len(self.pipes) == 0 or (SCREEN_WIDTH - self.pipes[-1]['x']) >= PIPE_DISTANCE:
            self.pipes.append({
                'x': SCREEN_WIDTH,
                'gap_y': random.randint(100, 300)
            })

        # Current pipe to pass
        current_pipe = self.pipes[0]
        pipe_x = current_pipe['x']
        pipe_gap_y = current_pipe['gap_y']

        done = False
        reward = 0.0

        # Update pipe rects
        top_pipe_rect = pygame.Rect(
            pipe_x, 0, PIPE_WIDTH, pipe_gap_y - PIPE_GAP // 2
        )
        bottom_pipe_rect = pygame.Rect(
            pipe_x,
            pipe_gap_y + PIPE_GAP // 2,
            PIPE_WIDTH,
            SCREEN_HEIGHT - (pipe_gap_y + PIPE_GAP // 2)
        )

        # Bird collision using masks
        rotated_bird = pygame.transform.rotate(self.bird_img, self.bird_angle)
        bird_mask = pygame.mask.from_surface(rotated_bird)
        bird_rect = rotated_bird.get_rect(center=(BIRD_X + BIRD_SIZE // 2, int(self.bird_y) + BIRD_SIZE // 2))

        # Pipe masks
        top_pipe_surface = pygame.Surface((PIPE_WIDTH, top_pipe_rect.height), pygame.SRCALPHA)
        top_pipe_surface.fill((0, 255, 0))
        top_pipe_mask = pygame.mask.from_surface(top_pipe_surface)

        bottom_pipe_surface = pygame.Surface((PIPE_WIDTH, bottom_pipe_rect.height), pygame.SRCALPHA)
        bottom_pipe_surface.fill((0, 255, 0))
        bottom_pipe_mask = pygame.mask.from_surface(bottom_pipe_surface)

        # Offsets for overlap
        top_offset = (top_pipe_rect.left - bird_rect.left, top_pipe_rect.top - bird_rect.top)
        bottom_offset = (bottom_pipe_rect.left - bird_rect.left, bottom_pipe_rect.top - bird_rect.top)

        collision_top = bird_mask.overlap(top_pipe_mask, top_offset)
        collision_bottom = bird_mask.overlap(bottom_pipe_mask, bottom_offset)

        # Collision or out of bounds
        if self.bird_y < 0 or self.bird_y + BIRD_SIZE > SCREEN_HEIGHT:
            done = True
            reward = -10
        elif collision_top or collision_bottom:
            done = True
            reward = -10
        elif pipe_x + PIPE_WIDTH < BIRD_X:
            self.score += 1
            reward = 1
            self.pipes.pop(0)  # Remove passed pipe
        else:
            reward = 0.1
            distance_to_gap = abs(self.bird_y - pipe_gap_y) / SCREEN_HEIGHT
            reward -= distance_to_gap * 0.05

        return self._get_state(), reward, done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Background
        self.window.blit(self.bg_img, (0, 0))

        if self.bird_vel < 0:
            self.bird_angle = max(-40, -self.bird_vel * 6)  # more upward
        else:
            self.bird_angle = min(40, -self.bird_vel * 2)  # slower fall tilt

        rotated_bird = pygame.transform.rotate(self.bird_img, self.bird_angle)

        # Get new rect centered at the bird's position
        bird_rect = rotated_bird.get_rect(center=(BIRD_X + BIRD_SIZE // 2, int(self.bird_y) + BIRD_SIZE // 2))
        self.window.blit(rotated_bird, bird_rect.topleft)
        self.bird_rect = bird_rect

        for pipe in self.pipes:
            # Top pipe
            top_height = pipe["gap_y"] - PIPE_GAP // 2
            top_img = pygame.transform.flip(self.pipe_img, False, True)
            top_rect = top_img.get_rect(bottomleft=(pipe["x"], top_height))
            self.window.blit(top_img, top_rect)

            # Bottom pipe
            bottom_y = pipe["gap_y"] + PIPE_GAP // 2
            bottom_rect = self.pipe_img.get_rect(topleft=(pipe["x"], bottom_y))
            self.window.blit(self.pipe_img, bottom_rect)

        # Score
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
