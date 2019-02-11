"""Objects that can be placed in a scene"""
from editor.scene.updatetype import UpdateType

class SceneObject():
    idCounter = 0

    @classmethod
    def incrementIdCounter(cls):
        ret = SceneObject.idCounter
        SceneObject.idCounter +=1
        return ret

    def __init__(self, name, scene):
        self.scene = scene
        self.name = name
        if scene:
            self.id = self.incrementIdCounter()

    def setName(self, name):
        self.name = name
        if self.scene:
            self.scene.update(UpdateType.OBJECT_NAME_CHANGE)
            self.scene.update(UpdateType.MATERIAL_NAME_CHANGE)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('scene')
        return d
    
    def __setstate__(self, d):
        self.__dict__ = d
        self.scene = None
