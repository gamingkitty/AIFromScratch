import pygame
from scratch_model import *
import sys
import numpy as np
import cupy as cp
from keras.datasets import cifar100


def load_cifar100(label_mode="fine"):
    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode=label_mode)

    train_images = train_images / 255
    test_images = test_images / 255

    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def main():
    black = (0, 0, 0)
    white = (255, 255, 255)
    dark_gray = (99, 102, 106)

    ai_model = Model.load("Models/cifar_100_v1_240_epochs")

    image = np.zeros((3, 32, 32))

    train_images, train_labels, test_images, test_labels = load_cifar100()
    current_image = 0

    pygame.init()
    pygame.event.set_allowed([
        pygame.KEYDOWN,
        pygame.QUIT,
        pygame.KEYUP,
        pygame.MOUSEBUTTONDOWN,
        pygame.MOUSEBUTTONUP
    ])

    scale = 20

    screen = pygame.display.set_mode((64 * scale + 300, 32 * scale))
    pygame.display.set_caption("CIFAR-100 Interface")

    info_rect = pygame.Rect((32 * scale, 0), (300, 32 * scale))

    font = pygame.font.SysFont("calibri", 28)
    title_font = pygame.font.SysFont("calibri", 36)

    info_top_text = title_font.render("Model Predictions", True, white)

    image_2 = np.zeros((3, 32, 32))

    clock = pygame.time.Clock()
    fps = 60

    mouse_held = False

    cifar100_labels = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm"
    ]

    while True:
        screen.fill(black)
        clock.tick(fps)

        pygame.draw.rect(screen, dark_gray, info_rect)
        screen.blit(
            info_top_text,
            (scale * 32 + (info_rect.width - info_top_text.get_width()) / 2, 20)
        )

        model_input = np.transpose(image, (0, 2, 1))[np.newaxis]
        predictions = cp.asnumpy(ai_model.predict(cp.array(model_input))[0])

        sorted_indices = np.argsort(predictions)[::-1]

        top_n = 10
        for i in range(top_n):
            label_index = sorted_indices[i]
            confidence = predictions[label_index]

            text = cifar100_labels[label_index] + f", {confidence * 100:0.2f}%"
            rendered_text = font.render(text, True, white)

            screen.blit(
                rendered_text,
                (scale * 32 + (info_rect.width - rendered_text.get_width()) / 2, 70 + i * 34)
            )

        true_label = cifar100_labels[test_labels[current_image]]
        rendered_text = font.render("True: " + true_label, True, white)
        screen.blit(
            rendered_text,
            (scale * 32 + (info_rect.width - rendered_text.get_width()) / 2, 450)
        )

        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                pygame.draw.rect(
                    screen,
                    (
                        int(255 * image[0, x, y]),
                        int(255 * image[1, x, y]),
                        int(255 * image[2, x, y])
                    ),
                    (x * scale, y * scale, scale, scale)
                )

        for y in range(image_2.shape[1]):
            for x in range(image_2.shape[2]):
                pygame.draw.rect(
                    screen,
                    (
                        int(255 * image_2[0, x, y]),
                        int(255 * image_2[1, x, y]),
                        int(255 * image_2[2, x, y])
                    ),
                    ((x + 32) * scale + 300, y * scale, scale, scale)
                )

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
                    current_image = min(current_image, len(test_images) - 1)

                    image = np.transpose(test_images[current_image], (0, 2, 1))

                    # z_data, a_data = ai_model.forward_propagate(
                    #     np.transpose(image, (0, 2, 1))[np.newaxis]
                    # )
                    #
                    # ai_model.backwards_propagate(
                    #     z_data,
                    #     a_data,
                    #     np.array([test_labels[current_image]])
                    # )

                    # f_dc_da = ai_model.final_dc_da[0]
                    # image_2 = np.transpose(f_dc_da, (0, 2, 1))

                    # mins = image_2.min()
                    # maxs = image_2.max()
                    # denom = maxs - mins
                    #
                    # image_2 = (image_2 - mins) / np.where(denom == 0, 1, denom)

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
                    add = image_2
                    image += add * 0.01

                    image = (image - image.min()) / (image.max() - image.min())

                    z_data, a_data = ai_model.forward_propagate(
                        np.transpose(image, (0, 2, 1))[np.newaxis]
                    )

                    ai_model.backwards_propagate(
                        z_data,
                        a_data,
                        np.array([test_labels[current_image]])
                    )

                    f_dc_da = ai_model.final_dc_da[0]
                    image_2 = np.transpose(f_dc_da, (0, 2, 1))

                    mins = image_2.min()
                    maxs = image_2.max()
                    denom = maxs - mins

                    image_2 = (image_2 - mins) / np.where(denom == 0, 1, denom)

        pygame.display.flip()


if __name__ == "__main__":
    main()