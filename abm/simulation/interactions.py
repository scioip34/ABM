"""Describing inter-entity interactions, such as agent-agent collisions, etc"""
import pygame


def within_group_collision(sprite1, sprite2):
    """Custom colllision check that omits collisions of sprite with itself. This way we can use group collision
    detect WITHIN a single group instead of between multiple groups"""
    if sprite1 != sprite2:
        return pygame.sprite.collide_circle(sprite1, sprite2)
    return False


def overlap(sprite1, sprite2):
    return sprite1.rect.colliderect(sprite2.rect)
