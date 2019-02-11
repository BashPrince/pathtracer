from editor.ui.ui_mainwindow import Ui_MainWindow
from editor.ui.ui_rendersettings_dialog import Ui_Dialog
from editor.ui.rendersettings import Rendersettings
from editor.ui.view import View
from editor.ui.viewport import Viewport
from editor.ui.attributeeditor import AttributeEditor
from editor.ui.materialeditor import MaterialEditor
from editor.scene.diffusematerial import DiffuseMaterial
from editor.scene.specularmaterial import SpecularMaterial
from editor.scene.scene import Scene
from editor.scene.cube import Cube
from editor.scene.sphere import Sphere
from editor.scene.plane import Plane
from editor.scene.disc import Disc
from editor.scene.camera import Camera
from editor.scene.light import Light
from editor.ui.scenelistview import SceneListView
from editor.ui.materiallistview import MaterialListView
from editor.ui.key_input_filter import KeyInputFilter
from editor.scene.updatetype import UpdateType

import sys
from PySide2.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QDialog
from PySide2.QtGui import QMatrix4x4, QVector3D, QVector4D


class MainWindow(View, QMainWindow):
    def __init__(self):
        View.__init__(self, None)
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.horizontalSplitter.setSizes([900, 500])
        self.ui.verticalSplitter.setSizes([400, 400])
        self.setWindowTitle('Renderer')

        self.scene = Scene()
        self.ui.actionNewCube.triggered.connect(self.createCube)
        self.ui.actionNewSphere.triggered.connect(self.createSphere)
        self.ui.actionNewPlane.triggered.connect(self.createPlane)
        self.ui.actionNewDisc.triggered.connect(self.createDisc)
        self.ui.actionNewDiffuseMaterial.triggered.connect(self.createDiffuseMaterial)
        self.ui.actionNewSpecularMaterial.triggered.connect(self.createSpecularMaterial)
        self.ui.actionNewLight.triggered.connect(self.createLight)
        self.ui.actionSave_Scene.triggered.connect(self.saveSceneSlot)
        self.ui.actionLoad_Scene.triggered.connect(self.loadSceneSlot)
        self.ui.actionNew_Scene.triggered.connect(self.newSceneSlot)
        self.ui.actionRender_Settings.triggered.connect(self.renderSettingsSlot)
        self.ui.actionSave_Image.triggered.connect(self.saveImageSlot)

        self.viewport = Viewport(self.scene, self.updateStatusBar)
        toDelete = self.ui.horizontalSplitter.replaceWidget(0, self.viewport)
        self.viewport.show()
        self.ui.actionRender.triggered.connect(self.renderSlot)
        toDelete.deleteLater()

        self.sceneListView = SceneListView(self.scene)
        toDelete = self.ui.horizontalLayout.replaceWidget(self.ui.sceneListWidget, self.sceneListView)
        self.sceneListView.show()
        toDelete.widget().deleteLater()

        self.materialListView = MaterialListView(self.scene)
        toDelete = self.ui.horizontalLayout_2.replaceWidget(self.ui.materialListWidget, self.materialListView)
        self.materialListView.show()
        toDelete.widget().deleteLater()

        self.attributeEditor = AttributeEditor(self.scene)
        toDelete = self.ui.verticalSplitter.replaceWidget(1, self.attributeEditor)
        self.attributeEditor.show()
        toDelete.deleteLater()

        self.materialEditor = MaterialEditor(self.scene)
        self.materialEditor.hide()

        self.activeEditor = self.attributeEditor

        self.objectKeyFilter = KeyInputFilter(self.scene.deleteSelectedObjects)
        self.materialKeyFilter = KeyInputFilter(self.scene.deleteSelectedMaterials)
        self.sceneListView.installEventFilter(self.objectKeyFilter)
        self.materialListView.installEventFilter(self.materialKeyFilter)

        self.scene.registerView(self, [UpdateType.OBJECT_SELECTION, UpdateType.MATERIAL_SELECTION])

    def notify(self, updateType):
        if updateType == UpdateType.OBJECT_SELECTION and not self.activeEditor is self.attributeEditor:
            self.ui.verticalSplitter.replaceWidget(1, self.attributeEditor)
            self.activeEditor = self.attributeEditor
            self.attributeEditor.show()
        elif updateType == UpdateType.MATERIAL_SELECTION and not self.activeEditor is self.materialEditor:
            self.ui.verticalSplitter.replaceWidget(1, self.materialEditor)
            self.activeEditor = self.materialEditor
            self.materialEditor.show()
    
    def createCube(self):
        self.scene.addObject(Cube('Cube', self.scene, self.scene.getDefaultMaterial()))

    def createSphere(self):
        self.scene.addObject(Sphere('Sphere', self.scene, self.scene.getDefaultMaterial()))

    def createPlane(self):
        self.scene.addObject(Plane('Plane', self.scene, self.scene.getDefaultMaterial()))

    def createDisc(self):
        self.scene.addObject(Disc('Disc', self.scene, self.scene.getDefaultMaterial()))
    
    def createDiffuseMaterial(self):
        self.scene.addMaterial(DiffuseMaterial('Diffuse Material', self.scene, QVector4D(0.7, 0.7, 0.7, 1.0)))

    def createSpecularMaterial(self):
        self.scene.addMaterial(SpecularMaterial('Specular Material', self.scene))
    
    def createLight(self):
        self.scene.addObject(Light('Light', self.scene))
    
    def newSceneSlot(self):
        self.scene.clear()
    
    def saveSceneSlot(self):
        fileName = QFileDialog.getSaveFileName(self, 'Save destination', dir="./testscenes")
        if fileName[0]:
            self.scene.saveScene(fileName[0])
    
    def loadSceneSlot(self):
        fileName = QFileDialog.getOpenFileName(self, 'Select file', dir="./testscenes")
        if fileName[0]:
            self.scene.loadScene(fileName[0])
    
    def saveImageSlot(self):
        self.viewport.saveImage()

    def renderSlot(self):
        if self.ui.actionRender.isChecked():
            self.viewport.startRender()
        else:
            self.viewport.stopRender()
    
    def closeEvent(self, event):
        self.viewport.closeViewport()
    
    def updateStatusBar(self, message):
        self.ui.statusbar.showMessage(message)
    
    def renderSettingsSlot(self):
        dialog = QDialog(self)
        dialog.ui = Ui_Dialog()
        dialog.ui.setupUi(dialog)

        renderSettings = self.viewport.getRenderSettings()
        dialog.ui.checkBoxLightSampling.setChecked(renderSettings.useLightSampling)
        dialog.ui.checkBoxCaustics.setChecked(renderSettings.renderCaustics)
        dialog.ui.spinBoxBounces.setValue(renderSettings.bounces)
        dialog.setWindowTitle('Render Settings')

        if dialog.exec_():
            self.viewport.setRenderSettings(Rendersettings(dialog.ui.checkBoxLightSampling.isChecked(), dialog.ui.checkBoxCaustics.isChecked(), dialog.ui.spinBoxBounces.value()))