from editor.scene.diffusematerial import DiffuseMaterial
from editor.scene.camera import Camera
from editor.scene.updatetype import UpdateType
from editor.scene.camera import Camera
from editor.scene.transformableobject import TransformableObject

import pickle


class Scene():
    """A 3D scene comprised of a number of arbitrarily positioned scene objects"""
    def __init__(self):
        self.objects = []
        self.materials = []
        self.selectedObjects = []
        self.selectedMaterials = []
        self.defaultMaterial = DiffuseMaterial('Default Material', self)
        self.views = {}
        self.camera = Camera('Camera', self)
        self.focusObject = TransformableObject('Focus Object', self)
        self.focusObject.setFocus(True)
        self.objects.append(self.camera)
    
    def registerView(self, view, updateTypeList):
        for upd in updateTypeList:
            self.views.setdefault(upd, []).append(view)


    def update(self, updateType):
        if updateType in self.views:
            for view in self.views[updateType]:
                view.notify(updateType)
        

    def addObject(self, newObject):
        self.objects.append(newObject)
        self.update(UpdateType.OBJECT_CREATE)

    def deleteObject(self, delObject):
        i = 0
        for obj in self.objects:
            if obj is delObject:
                if obj is self.focusObject:
                    self.unsetFocusObject()
                del self.objects[i]
                self.update(UpdateType.OBJECT_DELETE)
                return
            i += 1
    
    def deleteSelectedObjects(self):
        if not self.selectedObjects:
            return
        
        for sel in self.selectedObjects:
            i = 0
            for obj in self.objects:
                if obj is sel:
                    if obj is self.focusObject:
                        self.unsetFocusObject()
                    del self.objects[i]
                i += 1
        self.selectedObjects.clear()
        self.update(UpdateType.OBJECT_SELECTION)
        self.update(UpdateType.OBJECT_DELETE)
    
    def getObjectIterator(self):
        for obj in self.objects:
            yield obj

    def addMaterial(self, newMaterial):
        self.materials.append(newMaterial)
        self.update(UpdateType.MATERIAL_CREATE)

    def deleteMaterial(self, delMaterial):
        i = 0
        for mat in self.materials:
            if mat is delMaterial:
                for obj in self.objects:
                    obj.deleteMaterial(mat)
                del self.materials[i]
                self.update(UpdateType.MATERIAL_DELETE)
                return
            i += 1
    
    def deleteSelectedMaterials(self):
        if not self.selectedMaterials:
            return
        
        if self.defaultMaterial in self.selectedMaterials:
            return

        for sel in self.selectedMaterials:
            i = 0
            for mat in self.materials:
                if mat is sel:
                    for obj in self.objects:
                        obj.deleteMaterial(mat)
                    del self.materials[i]
                i += 1
        self.selectedMaterials.clear()
        self.update(UpdateType.MATERIAL_SELECTION)
        self.update(UpdateType.MATERIAL_DELETE)
    
    def getMaterialIterator(self):
        yield self.defaultMaterial
        for mat in self.materials:
            yield mat
    
    def getDefaultMaterial(self):
        return self.defaultMaterial
    
    def getCamera(self):
        return self.camera
    
    def setSelectedObjects(self, selectedObjects):
        self.selectedObjects = selectedObjects
        self.update(UpdateType.OBJECT_SELECTION)

    def setSelectedMaterials(self, selectedMaterials):
        self.selectedMaterials = selectedMaterials
        self.update(UpdateType.MATERIAL_SELECTION)

    def getSelectedObjectIterator(self):
        for obj in self.selectedObjects:
            yield obj
    
    def getSingleSelectedObject(self):
        if self.getNumSelectedObjects() == 1:
            return self.selectedObjects[0]
        else:
            return None
    
    def getSelectedMaterialIterator(self):
        for mat in self.selectedMaterials:
            yield mat
    
    def getNumSelectedObjects(self):
        return len(self.selectedObjects)
    
    def getNumSelectedMaterials(self):
        return len(self.selectedMaterials)

    def getFocusObject(self):
        return self.focusObject
    
    def setFocusObject(self, obj):
        if obj is self.focusObject:
            return
        
        self.focusObject.setFocus(False)
        self.focusObject = obj
        self.update(UpdateType.CAMERA_CHANGE)

    def unsetFocusObject(self):
        self.focusObject = TransformableObject('Focus Object', self)
        self.update(UpdateType.CAMERA_CHANGE)
    
    def clear(self):
        self.objects = []
        self.materials = []
        self.selectedObjects = []
        self.selectedMaterials = []
        self.camera = Camera('Camera', self)
        self.focusObject = TransformableObject('Focus Object', self)
        self.focusObject.setFocus(True)
        self.objects.append(self.camera)

        self.update(UpdateType.SCENE_LOAD)
        self.update(UpdateType.MATERIAL_CREATE)
        self.update(UpdateType.OBJECT_CREATE)
        self.update(UpdateType.OBJECT_SELECTION)
        self.update(UpdateType.MATERIAL_SELECTION)
    
    def saveScene(self, filename):
        with open(filename, 'wb') as file:
            sceneDict = {'camera':self.camera, 'focusObject':self.focusObject, 'defaultMaterial':self.defaultMaterial, 'objects':self.objects, 'materials':self.materials}
            pickle.dump(sceneDict, file)

    def loadScene(self, filename):
        with open(filename, 'rb') as file:
            sceneDict = pickle.load(file)
            self.camera = sceneDict['camera']
            self.focusObject = sceneDict['focusObject']
            self.defaultMaterial = sceneDict['defaultMaterial']
            self.objects = sceneDict['objects']
            self.materials = sceneDict['materials']
            self.selectedObjects = []
            self.selectedMaterials = []

            self.camera.scene = self
            self.defaultMaterial.scene = self

            for obj in self.objects:
                obj.scene = self
            
            for mat in self.materials:
                mat.scene = self

            self.update(UpdateType.SCENE_LOAD)
            self.update(UpdateType.MATERIAL_CREATE)
            self.update(UpdateType.OBJECT_CREATE)
            self.update(UpdateType.OBJECT_SELECTION)
            self.update(UpdateType.MATERIAL_SELECTION)