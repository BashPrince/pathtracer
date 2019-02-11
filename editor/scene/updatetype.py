from enum import Enum

class UpdateType(Enum):
    OBJECT_CREATE          = 1
    OBJECT_DELETE          = 2
    OBJECT_SELECTION       = 3
    OBJECT_TRANSFORM       = 4
    OBJECT_NAME_CHANGE     = 5
    MATERIAL_CREATE        = 6
    MATERIAL_DELETE        = 7
    MATERIAL_CHANGE        = 8
    MATERIAL_SELECTION     = 9
    MATERIAL_NAME_CHANGE   = 10
    LIGHT_CHANGE           = 11
    CAMERA_CHANGE          = 12
    SCENE_LOAD             = 13