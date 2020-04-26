from math import cos, pi
from noise import SimplexNoiseGen
from threading import Thread
from queue import Queue
from time import time
from random import randint
import config as Config


class ObjectManager(object):
    def __init__(self, ObjectClass):
        self.Object = {}
        self.ObjectClass = ObjectClass

    def GetObject(self, X):
        Obj = self.ObjectClass(X, self)
        return Obj


class CloudManager(ObjectManager):
    def __init__(self, CloudClass=None):
        self.Noise = SimplexNoiseGen(Config.Seed)

        if CloudClass == None:
            CloudClass = CloudChunk
        super(CloudManager, self).__init__(CloudClass)


class CloudChunk(object):
    def __init__(self, XPos, Generator):
        self.X = XPos
        self.Noise = Generator.Noise
        self.Generator = Generator

        self.Finished = False

        self.Generate()


#T = Thread(target=self.Generate)
#T.daemon = True
#T.start()

    def Generate(self):
        #print "Starting Generation at",self.X
        #start = time()
        Points = []
        Colours = []
        Length = 0

        YOffset = Config.CloudHeight / 2.0

        Noise = self.Noise
        NoiseOffset = Config.NoiseOffset

        for X in range(0, Config.CloudWidth):
            XOff = X + self.X

            for Y in range(0, Config.CloudHeight):
                Points.append(XOff)
                Points.append(Y)

                Colours.extend([1, 1, 1])
                #Get noise, round and clamp
                NoiseGen = Noise.fBm(XOff, Y) + NoiseOffset
                NoiseGen = max(0, min(1, NoiseGen))

                Colours.append(NoiseGen)

                Length += 1

        #Assign variables
        self.Points = Points
        self.Colours = Colours
        self.Length = Length

        #print "Finished Generation at", self.X
        #print "\tTook",time() - start
        self.Finished = True

    def Draw(self, X):
        if self.Finished:
            self.Finished = False
            self.GenerateFinshed()
