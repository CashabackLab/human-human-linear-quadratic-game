import data_visualization as dv
import matplotlib as mpl

wheel = dv.ColorWheel()


purples = ["#64256e", "#923aa1", "#c18bcc", "#ecdef0"][::-1]

darken_nums = [3, 1.3, 1.1, 1.1]
collapsed_condition_colors_light = [wheel.lighten_color(color,darken_num) for darken_num,color in zip(darken_nums,purples)]
lighten_nums = [2.25,1.,0.85,0.92]
collapsed_condition_colors_dark = [wheel.lighten_color(color,lighten_num) for lighten_num,color in zip(lighten_nums,purples)]

partner_color = wheel.lighten_color(wheel.grey,1.4) #wheel.blend_colors(wheel.light_grey, red)
self_color_light = wheel.dark_grey
self_color = wheel.lighten_color(wheel.grey, 0.7) #wheel.blend_colors(wheel.light_grey, wheel.jean_blue)

COLLAPSED_PLAYER_RELEVANCY = (
    # Self         Partner
    {"self":"irrelevant", "partner":"irrelevant"},
    {"self":"irrelevant", "partner":"relevant"},
    {"self":"relevant", "partner":"irrelevant"},
    {"self":"relevant", "partner":"relevant"},
)
collapsed_condition_names = ["Partner Irrelevant\nSelf Irrelevant", "Partner Relevant\nSelf Irrelevant",
                           "Partner Irrelevant\nSelf Relevant", "Partner Relevant\nSelf Relevant"]

model_names = [
    "No Partner Representation\nSelf Cost",
    "Partner Representation\nSelf Cost",
    "Partner Representation\nEqual Joint Cost",
    "Partner Representation\nWeighted Joint Cost",
]
model_names_threelines = [
    "No Partner\nRepresentation\nSelf Cost",
    "Partner\nRepresentation\nSelf Cost",
    "Partner\nRepresentation\nEqual Joint Cost",
    "Partner\nRepresentation\nWeighted Joint Cost",
]
