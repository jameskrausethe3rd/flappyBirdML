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

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.frames_since_last_flap = 0
        self.pipe_x = SCREEN_WIDTH
        self.pipe_gap_y = random.randint(100, 300)
        self.score = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.bird_y / SCREEN_HEIGHT,                      # Bird's vertical position (normalized)
            self.bird_vel / 10.0,                             # Bird's velocity (scaled)
            (self.pipe_x - 50) / SCREEN_WIDTH,                # Horizontal distance to pipe (normalized)
            (self.pipe_gap_y - self.bird_y) / SCREEN_HEIGHT,  # Vertical distance to gap center (normalized)
            (self.pipe_gap_y + PIPE_GAP / 2 - self.bird_y) / SCREEN_HEIGHT,  # Distance to bottom pipe opening
            (self.pipe_gap_y - PIPE_GAP / 2 - self.bird_y) / SCREEN_HEIGHT   # Distance to top pipe opening
        ], dtype=np.float32)

    def step(self, action):
        self.frames_since_last_flap += 1
        if action == 1 and self.frames_since_last_flap > 5:
            self.bird_vel = JUMP_STRENGTH
            self.frames_since_last_flap = 0

        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel
        self.pipe_x -= 2

        bird_rect = pygame.Rect(BIRD_X, self.bird_y, BIRD_SIZE, BIRD_SIZE)
        top_pipe_rect = pygame.Rect(
            self.pipe_x, 0, PIPE_WIDTH, self.pipe_gap_y - PIPE_GAP // 2
        )
        bottom_pipe_rect = pygame.Rect(
            self.pipe_x,
            self.pipe_gap_y + PIPE_GAP // 2,
            PIPE_WIDTH,
            SCREEN_HEIGHT - (self.pipe_gap_y + PIPE_GAP // 2)
        )

        done = False
        reward = 0.0

        if self.bird_y < 0 or self.bird_y + BIRD_SIZE > SCREEN_HEIGHT:
            done = True
            reward = -10
        elif bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
            done = True
            reward = -10
        elif self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = SCREEN_WIDTH
            self.pipe_gap_y = random.randint(100, 300)
            self.score += 1
            reward = 1
        else:
            # Survival reward
            reward = 0.1

            # Optional: shaping based on distance to pipe gap center
            gap_center = self.pipe_gap_y
            distance_to_gap = abs(self.bird_y - gap_center) / SCREEN_HEIGHT
            reward -= distance_to_gap * 0.05  # tweak weight as needed

        return self._get_state(), reward, done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.window.fill((135, 206, 235))  # Light blue background

        # Bird
        bird_rect = pygame.Rect(50, int(self.bird_y), 20, 20)
        pygame.draw.ellipse(self.window, (255, 255, 0), bird_rect)

        # Pipes
        pipe_width = 50
        top_pipe = pygame.Rect(self.pipe_x, 0, pipe_width, self.pipe_gap_y - PIPE_GAP // 2)
        bottom_pipe = pygame.Rect(self.pipe_x, self.pipe_gap_y + PIPE_GAP // 2, pipe_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.window, (34, 139, 34), top_pipe)
        pygame.draw.rect(self.window, (34, 139, 34), bottom_pipe)

        # Score
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
