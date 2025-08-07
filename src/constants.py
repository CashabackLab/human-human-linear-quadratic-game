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

cc_color = wheel.spearmint

COLLAPSED_PLAYER_RELEVANCY = (
    # Self         Partner
    {"self":"irrelevant", "partner":"irrelevant"},
    {"self":"irrelevant", "partner":"relevant"},
    {"self":"relevant", "partner":"irrelevant"},
    {"self":"relevant", "partner":"relevant"},
)

model_condition_names = ["both_irrelevant", "p1_irrelevant", "p1_relevant", "both_relevant"]
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
target_types = [
    "joint_irrelevant",
    "p1_relevant",
    "p2_relevant",
    "joint_relevant",
]
flipped_target_types = ["joint_irrelevant", "p2_relevant", "p1_relevant", "joint_relevant"]
PLAYER_RELEVANCY = [
    # Self         Partner
    {"p1":"irrelevant", "p2":"irrelevant"},
    {"p1":"relevant", "p2":"irrelevant"},
    {"p1":"irrelevant", "p2":"relevant"},
    {"p1":"relevant", "p2":"relevant"},
]
target_types_to_p1_p2_target_type = dict(zip(
    target_types, 
    PLAYER_RELEVANCY
))

target_x = 0.54
target_y = 0.3
rel_target_width = 0.01
irrel_target_width = 0.25
target_height = 0.01
rel_target_x_corner = target_x - 0.5*rel_target_width
rel_target_y_corner = target_y - 0.5*target_height
irrel_target_x_corner = target_x - 0.5*irrel_target_width
irrel_target_y_corner = target_y - 0.5*target_height