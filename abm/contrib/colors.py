def abs_color_to_float(color_tuple):
    """Given a color_tuple with R, G and B values between 0 and 255 we
    give back a touple with values between 0 and 1"""
    return tuple([comp/255 for comp in color_tuple])


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 109, 219)  # (0, 100, 255)
LIGHT_BLUE = (173, 216, 230)
GREEN = (26, 158, 26)  # (50, 150, 50)
PURPLE = (255, 109, 182)  # (130, 0, 130)
GREY = (230, 230, 230)
DARK_GREY = (210, 210, 210)
YELLOW = (190, 175, 50)
RED = (184, 0, 0)
LIGHT_RED = (255, 180, 180)
BACKGROUND = WHITE
