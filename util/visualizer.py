import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil
from PIL import Image
import glob

class Visualizer():

    def __init__(self):
        pass

    def hex_to_str(self, hexnum):
        zeros = '000000'
        hexstr = str(hex(hexnum))[2:]
        return '#' + (zeros[:6 - len(hexstr)]) + hexstr


    def make_binary_classiffication_frame(self, model, dataset, feature1, feature2, resolution, colors):
        colormap = {
            0: colors[0],
            1: colors[1],
        }
        xmin, xmax = dataset.get_min(feature1), dataset.get_max(feature1)
        ymin, ymax = dataset.get_min(feature2), dataset.get_max(feature2)
        
        xran = np.arange(xmin, xmax, ((xmax / resolution) - (xmin / resolution)))
        yran = np.arange(ymin, ymax, ((xmax / resolution) - (xmin / resolution)))

        curr = 0
        xpoints = []
        ypoints = []
        color = []
        
        for y in yran:
            for x in xran:
                temp = model.predict(np.array([[x], [y]]))[0][0]       
                curr = round(temp)
                xpoints.append(x)
                ypoints.append(y)
                color.append(self.hex_to_str(colormap[curr]))

        return xpoints, ypoints, color

    def plot_binary_classification(self, model, dataset, feature1, feature2, resolution=100, colors=(0x99d6ff, 0xffb3b3)):
        xpoints, ypoints, color = self.make_binary_classiffication_frame(model, dataset, feature1, feature2, resolution, colors)
        
        plt.scatter(xpoints, ypoints, s=10, c=color)
        dataset.plot()


    def animate_binary_classification(self, model, dataset, batch_size, epochs, feature1, feature2, resolution=100, colors=(0x99d6ff, 0xffb3b3), epoch_interval=5, ms_between_frames=50, ms_end_pause=1000):
        if os.path.exists('frames'):
            shutil.rmtree('frames')
        os.makedirs('frames')

        for i in range(epochs // epoch_interval):
            model.fit(dataset, batch_size=batch_size, epochs=epoch_interval)
            xpoints, ypoints, color = self.make_binary_classiffication_frame(model, dataset, feature1, feature2, resolution, colors)
            
            plt.scatter(xpoints, ypoints, s=10, c=color)
            dataset.plot(feature1, feature2, file_name=('frames/' + str(i) + '.png'), title=('Epoch: ' + str((i + 1) * epoch_interval)))
            plt.clf()

        for i in range(ms_end_pause // ms_between_frames):
            plt.scatter(xpoints, ypoints, s=10, c=color)
            dataset.plot(feature1, feature2, file_name=('frames/' + str((epochs // epoch_interval) + i) + '.png'), title=('Epoch: ' + str(epochs)))
            plt.clf()
        
        dataset.plot(feature1, feature2, file_name=('frames/' + str((epochs // epoch_interval) + i) + '.png'), title=('Epoch: ' + str(epochs)))
        plt.clf()

        frames = []
        images = sorted(glob.glob('frames/*.png'), key=os.path.getmtime) # out of order images?
        [print(i) for i in images]
        for img in images:
            frames.append(Image.open(img))

        frames[0].save('bin-class.gif', format='GIF', append_images=frames[1:], save_all=True, duration=ms_between_frames, loop=0)
        shutil.rmtree('frames')

