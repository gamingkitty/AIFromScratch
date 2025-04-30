import pygame
import model
import sys
import numpy as np
from keras.datasets import mnist


def load_data():
    (train_images_f, train_labels_f), (test_images_f, test_labels_f) = mnist.load_data()

    # Normalize the image data to values between 0 and 1
    train_images_f = train_images_f.astype('float32') / 255
    test_images_f = test_images_f.astype('float32') / 255

    # Flatten the images from 28x28 to 784-dimensional vectors
    train_images_f = train_images_f.reshape((-1, 28*28))
    test_images_f = test_images_f.reshape((-1, 28*28))

    return train_images_f, train_labels_f, test_images_f, test_labels_f


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

    ai_model = model.Model.load("Models/cross_entropy_test")

    image = np.zeros((28, 28))

    train_images, train_labels, test_images, test_labels = load_data()
    current_image = 0

    own_images = np.array([])
    own_labels = np.array([])

    # test_images = own_images
    # test_labels = own_labels

    # accuracy = ai_model.test(own_images, own_labels)

    # print(f"Accuracy on own numbers: {accuracy * 100}%")

    pygame.init()
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.QUIT, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

    scale = 20

    screen = pygame.display.set_mode((56 * scale + 300, 28 * scale))
    draw_rect = pygame.Rect((0, 0), (scale * 28, scale * 28))
    pygame.display.set_caption('Model Interface')

    info_rect = pygame.Rect((28 * scale, 0), (300, 28 * scale))

    font = pygame.font.SysFont("calibri", 36)

    info_top_text = font.render("Model Predictions", True, white)

    screen_width = screen.get_width()
    screen_height = screen.get_height()

    own_data = []
    own_labels = []

    image_2 = np.zeros((28, 28))

    clock = pygame.time.Clock()
    fps = 60

    mouse_held = False

    last_position = None

    create_num = None

    # Main Game Loop
    while True:
        screen.fill(black)
        delta_time = clock.tick(fps) / 1000

        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.rect(screen, dark_gray, info_rect)
        screen.blit(info_top_text, (scale * 28 + (info_rect.width - info_top_text.get_width()) / 2, 20))
        predictions = ai_model.predict(image.T.flatten())
        sorted_indices = np.argsort(predictions)[::-1]
        predictions = predictions[sorted_indices]

        for i in range(len(predictions)):
            text = str(sorted_indices[i]) + f", {predictions[i] * 100:0.2f}%"
            rendered_text = font.render(text, True, white)
            screen.blit(rendered_text, (scale * 28 + (info_rect.width - rendered_text.get_width()) / 2, 70 + i * 40))

        if mouse_pos[0] > 28 * scale + 300:
            mouse_pos = (mouse_pos[0] - (28 * scale + 300), mouse_pos[1])

        if mouse_held and draw_rect.collidepoint(mouse_pos):
            place_position = (mouse_pos[0] // scale, mouse_pos[1] // scale)

            if last_position is not None:
                x0, y0 = last_position
                x1, y1 = place_position

                num_steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
                xs = np.linspace(x0, x1, num_steps, dtype=int)
                ys = np.linspace(y0, y1, num_steps, dtype=int)

                for x, y in zip(xs, ys):
                    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                        image[x, y] += 0.4
                        image[x, y] = min(1, image[x, y])

                        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                                image[nx, ny] += 0.2
                                image[nx, ny] = min(1, image[nx, ny])

            last_position = place_position
        else:
            last_position = None

        for y in range(len(image)):
            for x in range(len(image[y])):
                brightness = image[x, y] ** (1 / 2.2)
                pygame.draw.rect(screen, (int(255 * brightness), int(255 * brightness), int(255 * brightness)), (x * scale, y * scale, scale, scale))

        for y in range(len(image_2)):
            for x in range(len(image_2[y])):
                pygame.draw.rect(screen, (int(255 * image_2[x, y]), int(255 * image_2[x, y]), int(255 * image_2[x, y])), ((x + 28) * scale + 300, y * scale, scale, scale))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # np.savez("Own Number Dataset/number_data.npz", images=own_data, labels=own_labels)
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_held = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_held = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    image = np.zeros((28, 28))
                elif event.key == pygame.K_w:
                    current_image += 1
                    image = test_images[current_image].reshape((28, 28)).T
                elif event.key == pygame.K_s:
                    current_image -= 1
                    current_image = max(current_image, 0)
                    image = test_images[current_image].reshape((28, 28)).T
                # elif pygame.K_0 <= event.key <= pygame.K_9:
                #     number_pressed = event.key - pygame.K_0
                #     ohe_number = np.zeros(10)
                #     ohe_number[number_pressed] = 1
                #     own_data.append(image.T.flatten())
                #     own_labels.append(ohe_number)
                #     image = np.zeros((28, 28))
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    number_pressed = event.key - pygame.K_0
                    create_num = number_pressed
                    label = np.zeros(10)
                    label[number_pressed] = 1
                    z_data, a_data = ai_model.forward_propagate(image.T)
                    ai_model.backwards_propagate(z_data, a_data, label)
                    image_2 = ai_model.final_dc_da
                    image_2 = (image_2 - image_2.min()) / (image_2.max() - image_2.min())
                    image_2 = image_2.reshape((28, 28)).T
                elif event.key == pygame.K_n:
                    noise = np.random.rand(784) - 0.5
                    noise *= 0.2
                    image += noise.reshape((28, 28))
                    image = (image - image.min()) / (image.max() - image.min())
                elif event.key == pygame.K_y:
                    add = 0.2 - image_2
                    add = np.maximum(add, 0)
                    image += add * 0.1

                    image = (image - image.min()) / (image.max() - image.min())

        if create_num is not None:
            number_pressed = create_num
            label = np.zeros(10)
            label[number_pressed] = 1
            z_data, a_data = ai_model.forward_propagate(image.T)
            ai_model.backwards_propagate(z_data, a_data, label)
            image_2 = ai_model.final_dc_da
            image_2 = (image_2 - image_2.min()) / (image_2.max() - image_2.min())
            image_2 = image_2.reshape((28, 28)).T

            image_2 = 0.3 - image_2
            image_2 = np.maximum(image_2, 0)
            image += image_2 * 0.5

            image = (image - image.min()) / (image.max() - image.min())

        # Update the screen
        pygame.display.flip()


if __name__ == "__main__":
    main()
