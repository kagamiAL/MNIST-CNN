import torch
import pygame
import os
from net import Net
from digit_recognition_trainer import normalize

model = Net().to("cuda")
model.load_state_dict(torch.load("./model-1-conv"))

pygame.init()
width, height = 28, 28
scale = 25
win = pygame.display.set_mode(size=(width * scale, height * scale))
screen = pygame.Surface((width, height))
clock = pygame.time.Clock()
black = (0, 0, 0)
white = (255, 255, 255)
running = True
drawing = False
erasing = False
t_canvas = torch.zeros(1, 784).to("cuda")


def color_canvas(color, x, y):
    if win.get_rect().collidepoint(x, y):
        WHITE = 255
        t_canvas[0][y // scale * width + x // scale] = WHITE if color == white else 0


def update_canvas():
    for j in range(t_canvas.shape[1]):
        if t_canvas[0][j] != 0:
            y = (j) // width
            x = (j) - y * width
            pygame.draw.rect(screen, white, (x, y, 1, 1))


def get_hashtags(val):
    return "#" * round(float(val) * 10)


def format_results(tensor: torch.Tensor):
    if not tensor.isnan().any():
        os.system("clear")
        output = ""
        for i in range(tensor.shape[1]):
            output += f"{i} : {get_hashtags(tensor[0][i])}\n"
        print(output)
        print(f"Result: {torch.argmax(tensor)}")


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            color_canvas(white, *event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                color_canvas(white, *event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                t_canvas.fill_(0)
            elif event.key == pygame.K_RETURN:
                format_results(model(normalize(t_canvas).view(-1, 28, 28)))
    screen.fill(black)
    update_canvas()
    win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0))
    pygame.display.flip()

pygame.quit()
