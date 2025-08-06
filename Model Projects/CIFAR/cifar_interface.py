import pygame
from scratch_model import model
import sys
import numpy as np
from keras.datasets import cifar10


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def main():
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (28, 128, 28)
    yellow = (230, 230, 0)
    brown = (118, 92, 72)
    gray = (175, 175, 175)
    dark_gray = (99, 102, 106)
    blue = (12, 246, 242)
    aqua = (5, 195, 221)
    red = (255, 0, 0)

    ai_model = model.Model.load("../Model Projects/Models/cifar_dense")

    image = np.zeros((3, 32, 32))

    train_images, train_labels, test_images, test_labels = load_cifar10()
    current_image = 0

    pygame.init()
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.QUIT, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

    scale = 20

    screen = pygame.display.set_mode((64 * scale + 300, 32 * scale))
    draw_rect = pygame.Rect((0, 0), (scale * 32, scale * 32))
    pygame.display.set_caption('Cifar Interface')

    info_rect = pygame.Rect((32 * scale, 0), (300, 32 * scale))

    font = pygame.font.SysFont("calibri", 36)

    info_top_text = font.render("Model Predictions", True, white)

    screen_width = screen.get_width()
    screen_height = screen.get_height()

    image_2 = np.zeros((3, 32, 32))

    clock = pygame.time.Clock()
    fps = 60

    mouse_held = False

    create_num = None

    cifar10_labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    # Main Game Loop
    while True:
        screen.fill(black)
        delta_time = clock.tick(fps) / 1000

        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.rect(screen, dark_gray, info_rect)
        screen.blit(info_top_text, (scale * 32 + (info_rect.width - info_top_text.get_width()) / 2, 20))
        predictions = ai_model.predict(np.transpose(image, (0, 2, 1)))
        sorted_indices = np.argsort(predictions)[::-1]
        predictions = predictions[sorted_indices]

        for i in range(len(predictions)):
            text = cifar10_labels[sorted_indices[i]] + f", {predictions[i] * 100:0.2f}%"
            rendered_text = font.render(text, True, white)
            screen.blit(rendered_text, (scale * 32 + (info_rect.width - rendered_text.get_width()) / 2, 70 + i * 40))

        rendered_text = font.render(cifar10_labels[test_labels[current_image]], True, white)
        screen.blit(rendered_text, (scale * 32 + (info_rect.width - rendered_text.get_width()) / 2, 130 + len(predictions) * 40))

        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                pygame.draw.rect(screen, (int(255 * image[0, x, y]), int(255 * image[1, x, y]), int(255 * image[2, x, y])), (x * scale, y * scale, scale, scale))

        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                pygame.draw.rect(screen, (int(255 * image_2[0, x, y]), int(255 * image_2[1, x, y]), int(255 * image_2[2, x, y])), ((x + 32) * scale + 300, y * scale, scale, scale))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_held = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_held = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    image = np.zeros((3, 32, 32))
                elif event.key == pygame.K_w:
                    current_image += 1
                    image = np.transpose(test_images[current_image], (0, 2, 1))
                    z_data, a_data = ai_model.forward_propagate(np.transpose(image, (0, 2, 1)))
                    ai_model.backwards_propagate(z_data, a_data, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 1)
                    image_2 = np.transpose(ai_model.final_dc_da.reshape(3, 32, 32), (0, 2, 1))
                    image_2 = (image_2 - image_2.min()) / (image_2.max() - image_2.min())
                elif event.key == pygame.K_s:
                    current_image -= 1
                    current_image = max(current_image, 0)
                    image = np.transpose(test_images[current_image], (0, 2, 1))
                elif event.key == pygame.K_n:
                    noise = np.random.rand(3, 32, 32) - 0.5
                    noise *= 0.2
                    image += noise.reshape((3, 32, 32))
                    image = (image - image.min()) / (image.max() - image.min())
                elif event.key == pygame.K_y:
                    add = 0.2 - image_2
                    add = np.maximum(add, 0)
                    image += add * 0.1

                    image = (image - image.min()) / (image.max() - image.min())

        # Update the screen
        pygame.display.flip()


if __name__ == "__main__":
    main()
