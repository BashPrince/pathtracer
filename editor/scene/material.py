from editor.scene.updatetype import UpdateType
from editor.scene.sceneobject import SceneObject
from PySide2.QtGui import QVector4D

class Material(SceneObject):
    def __init__(self, name, scene, editorColor):
        super().__init__(name, scene)
        self.editorColor = editorColor
    
    def getEditorColor(self):
        return self.editorColor