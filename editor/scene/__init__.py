__all__ = ['camera', 'cube', 'sphere', 'plane', 'disc', 'material', 'scene', 'sceneobject', 'transformableobject', 'updatetype']

from editor.scene.cube import Cube
from editor.scene.sphere import Sphere
from editor.scene.plane import Plane
from editor.scene.disc import Disc
from editor.scene.viewportplane import ViewportPlane
from enum import IntEnum

class SceneObjectType(IntEnum):
    SPHERE = 0
    CUBE   = 1
    PLANE  = 2
    DISC   = 3
    CAMERA = 4

shapeDict = {SceneObjectType.SPHERE:"Sphere", SceneObjectType.CUBE:"Cube", SceneObjectType.PLANE:"Plane", SceneObjectType.DISC:"Disc"}
drawableClasses = [Cube, Sphere, Plane, Disc, ViewportPlane]