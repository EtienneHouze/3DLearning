from __future__ import absolute_import, print_function, division

class Point:

    def __init__(self, *args):
        self.x = 0
        self.y = 0
        self.z = 0
        self.label = 0
        if len(args) != 3:
            pass
        self.x = float(args[0]); self.y = float(args[1]); self.z = float(args[2])

    def __str__(self):
        return str(self.x)+" "+str(self.y)+" "+str(self.z)+" "+str(self.label)