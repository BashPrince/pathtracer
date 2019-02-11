from editor.ui.ui_materialeditor import Ui_MaterialEditor
from editor.scene.updatetype import UpdateType
from editor.scene.diffusematerial import DiffuseMaterial
from editor.scene.specularmaterial import SpecularMaterial
from editor.ui.view import View

from PySide2.QtWidgets import QWidget, QColorDialog
from PySide2.QtGui import QMatrix4x4, QVector3D, QVector4D, QColor
from PySide2 import QtCore


class MaterialEditor(View, QWidget):
    def __init__(self, scene, parent = None):
        View.__init__(self, scene)
        QWidget.__init__(self, parent)
        self.ui = Ui_MaterialEditor()
        self.ui.setupUi(self)
        self.ui.colorWidget1.hide()
        self.ui.colorWidget2.hide()
        self.ui.IORwidget.hide()
        self.ui.colorSwatch1.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.ui.colorSwatch2.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self.connectAll()

        self.scene.registerView(self, [UpdateType.MATERIAL_SELECTION, UpdateType.MATERIAL_NAME_CHANGE])
        self.notify(UpdateType.MATERIAL_SELECTION)
    
    def notify(self, updateType):
        self.ui.name.setText('No materials selected')
        self.ui.colorWidget1.hide()
        self.ui.colorWidget2.hide()
        self.ui.IORwidget.hide()

        if self.scene.getNumSelectedMaterials() > 1:
            self.ui.name.setText('Multiple materials selected')
            return

        for sel in self.scene.getSelectedMaterialIterator():
            if type(sel) == DiffuseMaterial:
                self.setupDiffuseMaterial(sel)
            elif type(sel) == SpecularMaterial:
                self.setupSpecularMaterial(sel)
            break
    
    def colorSlot1(self, value):
        if self.scene.getNumSelectedMaterials() != 1:
            return
        
        for mat in self.scene.getSelectedMaterialIterator():
            colorDialog = QColorDialog(parent = self)
            color = colorDialog.getColor(options = QColorDialog.ShowAlphaChannel)
            if color.isValid():
                if type(mat) == DiffuseMaterial:
                    mat.setColor(QVector4D(color.redF(), color.greenF(), color.blueF(), color.alphaF()))
                    self.setupDiffuseMaterial(mat)
                elif type(mat) == SpecularMaterial:
                    mat.setReflectionColor(QVector4D(color.redF(), color.greenF(), color.blueF(), color.alphaF()))
                    self.setupSpecularMaterial(mat)
            return

    def colorSlot2(self, value):
        if self.scene.getNumSelectedMaterials() != 1:
            return
        
        for mat in self.scene.getSelectedMaterialIterator():
            colorDialog = QColorDialog(parent = self)
            color = colorDialog.getColor(options = QColorDialog.ShowAlphaChannel)
            if color.isValid() and type(mat) == SpecularMaterial:
                    mat.setTransmissionColor(QVector4D(color.redF(), color.greenF(), color.blueF(), color.alphaF()))
                    self.setupSpecularMaterial(mat)
            return
    
    def IORSlot(self, value):
        if self.scene.getNumSelectedMaterials() != 1:
            return
        
        for mat in self.scene.getSelectedMaterialIterator():
            if type(mat) == SpecularMaterial:
                mat.setIOR(value)
                return
    
    def connectAll(self):
        self.ui.colorPushButton1.clicked.connect(self.colorSlot1)
        self.ui.colorPushButton2.clicked.connect(self.colorSlot2)
        self.ui.iorSpinBox.valueChanged.connect(self.IORSlot)


    def disconnectAll(self):
        self.ui.colorPushButton1.clicked.disconnect(self.colorSlot1)
        self.ui.colorPushButton2.clicked.disconnect(self.colorSlot2)
        self.ui.iorSpinBox.valueChanged.disconnect(self.IORSlot)
    
    def setUiColor(self, color1, color2 = None):
        self.ui.labelColor1.setText("({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(color1.x(), color1.y(), color1.z(), color1.w()))
        self.ui.colorSwatch1.setStyleSheet('background-color: rgb({}, {}, {}, {});'.format(
            int(color1.x() * 255), int(color1.y() * 255), int(color1.z() * 255), int(color1.w() * 255)))
        
        if color2:
            self.ui.labelColor2.setText("({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(color2.x(), color2.y(), color2.z(), color2.w()))
            self.ui.colorSwatch2.setStyleSheet('background-color: rgb({}, {}, {}, {});'.format(
                int(color2.x() * 255), int(color2.y() * 255), int(color2.z() * 255), int(color2.w() * 255)))
    
    def setupDiffuseMaterial(self, diffuseMat):
        self.ui.name.setText(diffuseMat.name)
        self.disconnectAll()
        self.ui.colorWidget1.show()
        self.ui.labelName1.setText('Diffuse Color:')
        if diffuseMat is self.scene.getDefaultMaterial():
            self.ui.colorPushButton1.setEnabled(False)
        else:
            self.ui.colorPushButton1.setEnabled(True)
        self.setUiColor(diffuseMat.getColor())
        self.connectAll()

    def setupSpecularMaterial(self, specularMat):
        self.ui.name.setText(specularMat.name)
        self.disconnectAll()
        self.ui.colorWidget1.show()
        self.ui.colorWidget2.show()
        self.ui.IORwidget.show()
        self.ui.labelName1.setText('Reflection Color:')
        self.ui.labelName2.setText('Transmission Color:')
        if specularMat is self.scene.getDefaultMaterial():
            self.ui.colorPushButton1.setEnabled(False)
            self.ui.colorPushButton2.setEnabled(False)
        else:
            self.ui.colorPushButton1.setEnabled(True)
            self.ui.colorPushButton2.setEnabled(True)
        self.setUiColor(specularMat.getReflectionColor(), specularMat.getTransmissionColor())
        self.ui.iorSpinBox.setValue(specularMat.getIOR())
        self.connectAll()