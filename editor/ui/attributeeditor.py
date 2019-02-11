from editor.ui.ui_attributeeditor import Ui_AttributeEditor
from editor.scene.updatetype import UpdateType
from editor.scene.light import Light
from editor.scene import shapeDict
from editor.ui.view import View

from PySide2.QtWidgets import QWidget, QColorDialog
from PySide2.QtGui import QMatrix4x4, QVector3D, QVector4D
from PySide2 import QtCore


class AttributeEditor(View, QWidget):
    

    def __init__(self, scene, parent = None):
        View.__init__(self, scene)
        QWidget.__init__(self, parent)
        self.ui = Ui_AttributeEditor()
        self.ui.setupUi(self)
        self.ui.colorSwatch.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.ui.translationWidget.hide()
        self.ui.scaleWidget.hide()
        self.ui.rotationWidget.hide()
        self.ui.focusCheckBox.hide()
        self.ui.materialWidget.hide()
        self.ui.lightWidget.hide()
        self.ui.cameraWidget.hide()

        for k in shapeDict:
                self.ui.shapeComboBox.addItem(shapeDict[k], k)

        self.connectAll()

        self.scene.registerView(self, [UpdateType.OBJECT_SELECTION, UpdateType.OBJECT_NAME_CHANGE, UpdateType.MATERIAL_CREATE, UpdateType.MATERIAL_DELETE, UpdateType.LIGHT_CHANGE, UpdateType.MATERIAL_NAME_CHANGE])
        self.ui.name.setText('No objects selected')
        self.notify(UpdateType.MATERIAL_CREATE)
    
    def notify(self, updateType):
        if updateType in [UpdateType.MATERIAL_CREATE, UpdateType.MATERIAL_DELETE, UpdateType.MATERIAL_NAME_CHANGE]:
            self.disconnectAll()
            self.ui.materialComboBox.clear()

            selectedObj = self.scene.getSingleSelectedObject()
            for mat in self.scene.getMaterialIterator():
                self.ui.materialComboBox.addItem(mat.name, mat)
            if selectedObj:
                self.setMaterialIndex(selectedObj)

            self.connectAll()
        elif updateType in [UpdateType.OBJECT_SELECTION, UpdateType.OBJECT_NAME_CHANGE, UpdateType.LIGHT_CHANGE]:
            self.ui.name.setText('No objects selected')
            self.ui.translationWidget.hide()
            self.ui.scaleWidget.hide()
            self.ui.rotationWidget.hide()
            self.ui.focusCheckBox.hide()
            self.ui.materialWidget.hide()
            self.ui.cameraWidget.hide()
            self.ui.lightWidget.hide()

            if self.scene.getNumSelectedObjects() > 1:
                self.ui.name.setText('Multiple objects selected')
                return

            for sel in self.scene.getSelectedObjectIterator():
                self.ui.name.setText(sel.name)
                self.disconnectAll()
                if sel is not self.scene.getCamera():
                    self.ui.translationWidget.show()
                    self.ui.scaleWidget.show()
                    self.ui.rotationWidget.show()
                    self.ui.focusCheckBox.show()

                    if isinstance(sel, Light):
                        self.ui.lightWidget.show()
                        self.ui.lightSpinbox.setValue(sel.getIntensity())
                        self.setUiColor(sel.getColor())
                        self.ui.shapeComboBox.setCurrentIndex(sel.getEmitterShape())
                        self.ui.checkBoxLightVisible.setChecked(sel.getVisibility())
                    else:
                        self.ui.materialWidget.show()

                else:
                    self.ui.cameraWidget.show()
                    self.ui.fStopSpinbox.setValue(sel.getFstop())
                    self.ui.focusDistanceSpinbox.setValue(sel.getFocusDistance())
                    self.ui.stratificationSpinbox.setValue(sel.getStratificationLevel())

                self.setUiTranslation(sel.getTranslation())
                self.setUiScale(sel.getScale())
                self.setUiRotation(sel.getRotation())
                self.ui.focusCheckBox.setChecked(sel.isFocusObject())

                
                self.setMaterialIndex(sel)

                self.connectAll()
                break
    
    def setMaterialIndex(self, selectedObj):
        for index in range(0, self.ui.materialComboBox.count()):
            itemAtIndex = self.ui.materialComboBox.itemData(index)
            if itemAtIndex is selectedObj.material:
                self.ui.materialComboBox.setCurrentIndex(index)
                break
            if itemAtIndex is self.scene.getDefaultMaterial():
                self.ui.materialComboBox.setCurrentIndex(index)
    
    def materialSlot(self, index):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            mat = self.ui.materialComboBox.itemData(index)
            if not mat is obj.material:
                obj.setMaterial(mat)
            return
    
    def translationSlot(self, value):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            obj.setTranslation(QVector3D(self.ui.xTranslationSpinbox.value(), self.ui.yTranslationSpinbox.value(), self.ui.zTranslationSpinbox.value()))
            return
    
     
    def scaleSlot(self, value):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            obj.setScale(QVector3D(self.ui.xScaleSpinbox.value(), self.ui.yScaleSpinbox.value(), self.ui.zScaleSpinbox.value()))
            return
    
    def rotateSlot(self, value):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            obj.setRotation(QVector3D(self.ui.xRotationSpinbox.value(), self.ui.yRotationSpinbox.value(), self.ui.zRotationSpinbox.value()))
            return
    
    def focusSlot(self, focus):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            obj.userSetFocus(True if focus else False)
            return
    
    def intensitySlot(self, intensity):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            obj.setIntensity(intensity)
            return
    
    def colorSlot(self, value):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for light in self.scene.getSelectedObjectIterator():
            colorDialog = QColorDialog(parent = self)
            color = colorDialog.getColor(options = QColorDialog.ShowAlphaChannel)
            if color.isValid():
                light.setColor(QVector4D(color.redF(), color.greenF(), color.blueF(), color.alphaF()))
            return
    
    def shapeSlot(self, index):
        if self.scene.getNumSelectedObjects() != 1:
            return

        for obj in self.scene.getSelectedObjectIterator():
            shapeType = self.ui.shapeComboBox.itemData(index)
            if not shapeType is obj.emitterShape:
                obj.setEmitterShape(shapeType)
            return
    
    def visibilitySlot(self, isVisible):
        if self.scene.getNumSelectedObjects() != 1:
            return
        
        for obj in self.scene.getSelectedObjectIterator():
            obj.setVisibility(True if isVisible else False)
            return
    
    def focusDistanceSlot(self, focusDistance):
        self.scene.getCamera().setFocusDistance(focusDistance)
    
    def fStopSlot(self, fStop):
        self.scene.getCamera().setFstop(fStop)
    
    def stratificationSlot(self, level):
        self.scene.getCamera().setStratificationLevel(level)

    def connectAll(self):
        self.ui.xTranslationSpinbox.valueChanged.connect(self.translationSlot)
        self.ui.yTranslationSpinbox.valueChanged.connect(self.translationSlot)
        self.ui.zTranslationSpinbox.valueChanged.connect(self.translationSlot)

        self.ui.xScaleSpinbox.valueChanged.connect(self.scaleSlot)
        self.ui.yScaleSpinbox.valueChanged.connect(self.scaleSlot)
        self.ui.zScaleSpinbox.valueChanged.connect(self.scaleSlot)
        
        self.ui.xRotationSpinbox.valueChanged.connect(self.rotateSlot)
        self.ui.yRotationSpinbox.valueChanged.connect(self.rotateSlot)
        self.ui.zRotationSpinbox.valueChanged.connect(self.rotateSlot)

        self.ui.focusCheckBox.stateChanged.connect(self.focusSlot)
        self.ui.materialComboBox.currentIndexChanged.connect(self.materialSlot)

        self.ui.fStopSpinbox.valueChanged.connect(self.fStopSlot)
        self.ui.focusDistanceSpinbox.valueChanged.connect(self.focusDistanceSlot)
        self.ui.stratificationSpinbox.valueChanged.connect(self.stratificationSlot)

        self.ui.lightSpinbox.valueChanged.connect(self.intensitySlot)
        self.ui.colorPushButton.clicked.connect(self.colorSlot)
        self.ui.shapeComboBox.currentIndexChanged.connect(self.shapeSlot)
        self.ui.checkBoxLightVisible.stateChanged.connect(self.visibilitySlot)


    def disconnectAll(self):
        self.ui.xTranslationSpinbox.valueChanged.disconnect(self.translationSlot)
        self.ui.yTranslationSpinbox.valueChanged.disconnect(self.translationSlot)
        self.ui.zTranslationSpinbox.valueChanged.disconnect(self.translationSlot)

        self.ui.xScaleSpinbox.valueChanged.disconnect(self.scaleSlot)
        self.ui.yScaleSpinbox.valueChanged.disconnect(self.scaleSlot)
        self.ui.zScaleSpinbox.valueChanged.disconnect(self.scaleSlot)

        self.ui.xRotationSpinbox.valueChanged.disconnect(self.rotateSlot)
        self.ui.yRotationSpinbox.valueChanged.disconnect(self.rotateSlot)
        self.ui.zRotationSpinbox.valueChanged.disconnect(self.rotateSlot)

        self.ui.focusCheckBox.stateChanged.disconnect(self.focusSlot)
        self.ui.materialComboBox.currentIndexChanged.disconnect(self.materialSlot)

        self.ui.fStopSpinbox.valueChanged.disconnect(self.fStopSlot)
        self.ui.focusDistanceSpinbox.valueChanged.disconnect(self.focusDistanceSlot)
        self.ui.stratificationSpinbox.valueChanged.disconnect(self.stratificationSlot)

        self.ui.lightSpinbox.valueChanged.disconnect(self.intensitySlot)
        self.ui.colorPushButton.clicked.disconnect(self.colorSlot)
        self.ui.shapeComboBox.currentIndexChanged.disconnect(self.shapeSlot)
        self.ui.checkBoxLightVisible.stateChanged.disconnect(self.visibilitySlot)


    def setUiTranslation(self, translationVec):
        self.ui.xTranslationSpinbox.setValue(translationVec.x())
        self.ui.yTranslationSpinbox.setValue(translationVec.y())
        self.ui.zTranslationSpinbox.setValue(translationVec.z())

    def setUiRotation(self, rotationVec):
        self.ui.xRotationSpinbox.setValue(rotationVec.x())
        self.ui.yRotationSpinbox.setValue(rotationVec.y())
        self.ui.zRotationSpinbox.setValue(rotationVec.z())

    def setUiScale(self, scaleVec):
        self.ui.xScaleSpinbox.setValue(scaleVec.x())
        self.ui.yScaleSpinbox.setValue(scaleVec.y())
        self.ui.zScaleSpinbox.setValue(scaleVec.z())
    
    def setUiColor(self, color):
        self.ui.labelRGBA.setText("({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(color.x(), color.y(), color.z(), color.w()))
        self.ui.colorSwatch.setStyleSheet('background-color: rgb({}, {}, {}, {});'.format(
            int(color.x() * 255), int(color.y() * 255), int(color.z() * 255), int(color.w() * 255)))
