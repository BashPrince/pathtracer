class Rendersettings():
    def __init__(self, useLightSampling = True, renderCaustics = True, bounces = 30):
        self.useLightSampling = useLightSampling
        self.renderCaustics = renderCaustics
        self.bounces = bounces if bounces >= 0 else 0
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__