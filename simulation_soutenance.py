from simple_playgrounds import Engine
from simple_playgrounds.playgrounds import SingleRoom
from simple_playgrounds.entities.texture import UniformTexture, CenteredRandomTilesTexture, PolarStripesTexture, RandomTilesTexture
from simple_playgrounds.utils import PositionAreaSampler
from simple_playgrounds.entities.scene_elements import Basic
from simple_playgrounds.entities.agents import BaseAgent, HeadAgent
from simple_playgrounds.entities.agents.sensors import DistanceArraySensor
from simple_playgrounds.controllers import Keyboard, Random
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

SHOW_ENVIRONMENT = True
SHOW_HEATMAPS = False
SAVE_SIMU = False
SCALE = 1.5

my_playground = SingleRoom(size=(int(250*SCALE), int(250*SCALE)), wall_type='light')
# my_agent = HeadAgent(controller=Keyboard())
my_agent = BaseAgent(name='robot', controller=Random())

IR_sensor = DistanceArraySensor(my_agent.base_platform, name='IR_1', normalize=True, range=int(250*SCALE),
                                fov=270, number=1080, point_angle=0.125)
my_agent.add_sensor(IR_sensor)

engine = Engine(time_limit=100, agents=my_agent, playground=my_playground, screen=SHOW_ENVIRONMENT)

##################################
# AMENAGEMENT DE L'ENVIRONNEMENT #
##################################

text_tiles_circle = RandomTilesTexture(color_min=(0, 0, 150), color_max=(0, 0, 255), size_tiles=int(5*SCALE),
                                       radius=int(40*SCALE))
circular_object = Basic((int(205*SCALE), int(90*SCALE), 0), physical_shape='circle', radius=int(35*SCALE),
                        texture=text_tiles_circle)
my_playground.add_scene_element(circular_object)


text_uniform_rectangle = UniformTexture(color_min=(255, 50, 0), color_max=(255, 150, 0), size_tiles=int(1*SCALE),
                                        radius=int(40*SCALE))
rectangular_object1 = Basic((int(40*SCALE), int(100*SCALE), 0.15), physical_shape='rectangle',
                            width_length=(int(15*SCALE), int(60*SCALE)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object2 = Basic((int(50*SCALE), int(40*SCALE), 1.2), physical_shape='rectangle',
                            width_length=(int(30*SCALE), int(50*SCALE)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object2)

text_random_tiles_centered = CenteredRandomTilesTexture(color_min=(255, 0, 100), color_max=(255, 130, 255),
                                                        radius=30*SCALE, size_tiles=int(10*SCALE))
square_object = Basic([int(80*SCALE), int(180*SCALE), 0.25], physical_shape='square', radius=int(30*SCALE),
                      texture=text_random_tiles_centered)
my_playground.add_scene_element(square_object)

pentagonal_object = Basic((int(200*SCALE), int(40*SCALE), 0.2), default_config_key='pentagon', radius=int(30*SCALE))
my_playground.add_scene_element(pentagonal_object)

hexagonal_object = Basic((int(67*SCALE), int(85*SCALE), 0.2), default_config_key='hexagon', radius=int(30*SCALE))
my_playground.add_scene_element(hexagonal_object)

text_uniform_triangle = UniformTexture(color_min=(255, 255, 0), color_max=(0, 255, 0), size_tiles=int(1*SCALE),
                                       radius=int(20*SCALE))
center_area, size_area = my_playground.get_quarter_area((0, 0), 'up-right')
area_all = PositionAreaSampler(center=center_area, area_shape='circle', radius=int(size_area[0]/4*SCALE))
n_triangles = 3
for i in range(n_triangles):
    circular_object = Basic(area_all, physical_shape='triangle', radius=int(20*SCALE), texture=text_uniform_triangle,
                            allow_overlapping=False)
    my_playground.add_scene_element(circular_object)


#################################
# SIMULATION ET RECOLTE DONNEES #
#################################

all_observations = {}
all_actions = {}
for agent in engine.agents:
    all_observations[agent.name] = {}
    all_actions[agent.name] = {}
    for part in agent.parts:
        all_actions[agent.name][part.name] = {}

if SHOW_HEATMAPS:
    heatMat = np.zeros((my_playground.length, my_playground.width, 7))

while engine.game_on:

    if SHOW_HEATMAPS:
        i, j, k = int(agent.position[0]), int(agent.position[1]), int(agent.position[2])
        heatMat[i, j, k] += 1

    print(str(engine.total_elapsed_time) + '/' + str(engine.time_limit))

    if SHOW_ENVIRONMENT:
        engine.display_full_scene()

    actions = {}
    for agent in engine.agents:
        actions[agent.name] = agent.controller.generate_actions()
        if engine.total_elapsed_time == 0 and SAVE_SIMU:
            for part in agent.parts:
                available_actions = part.get_available_actions()
                for action in available_actions:
                    all_actions[agent.name][part.name][action.action.name] = [actions[agent.name][
                                                                                  part.name][action.action]]
        elif engine.total_elapsed_time > 0 and SAVE_SIMU:
            for part in agent.parts:
                available_actions = part.get_available_actions()
                for action in available_actions:
                    all_actions[agent.name][part.name][action.action.name].append(actions[agent.name][
                                                                                      part.name][action.action])

    if engine.total_elapsed_time == 0:  # Pour contourner un petit bug temporaire
        engine.step(actions)
    else:
        engine.multiple_steps(actions, n_steps=2)

    if SAVE_SIMU or SHOW_ENVIRONMENT:
        engine.update_observations()
        observation = IR_sensor.sensor_value

    if SAVE_SIMU:
        for agent in engine.agents:
            for sensor in agent.sensors:
                observation = sensor.sensor_value
                sensor_name = sensor.name
                if engine.total_elapsed_time == 1:
                    all_observations[agent.name][sensor_name] = [observation]
                else:
                    all_observations[agent.name][sensor_name].append(observation)

    if SHOW_ENVIRONMENT:
        cv2.imshow('sensor', engine.generate_sensor_image(my_agent))
        # cv2.waitKey(20)

    if (engine.total_elapsed_time-1) % 10000 == 0 and SHOW_HEATMAPS:
        heatMatAllDirections = np.zeros((my_playground.length, my_playground.width))
        heatMatCumulative = np.zeros((my_playground.length, my_playground.width))
        for theta in range(np.shape(heatMat)[2]):
            heatMatAllDirections += heatMat[:, :, theta] > 0 * 1
            # heatMatCumulative += heatMat[:, :, theta]
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.rot90(heatMatCumulative), cmap='gray')
        # plt.title("Cumul total")
        # plt.subplot(1, 2, 2)
        plt.imshow(np.rot90(heatMatAllDirections), cmap='gray')
        plt.title("$T="+str(int((engine.total_elapsed_time)/2))+str('$'))
        plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.98)
        plt.axis('off')
        # plt.savefig("HeatMap_STEP="+str(int((engine.total_elapsed_time-1)/2)),dpi=450)
        plt.show()

engine.terminate()
cv2.destroyAllWindows()

if SAVE_SIMU:
    fid = open(
        "data/recordings/environment_soutenance/IR_data_fov270_1080dots_60000stepsMultiStep=2_soutenance_scale13E-1.p", "wb")
    pickle.dump({'observations': all_observations, 'actions': all_actions}, fid)
