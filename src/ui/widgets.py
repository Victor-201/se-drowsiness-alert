from kivy.uix.button import Button
from kivy.uix.image import Image


class IconButton(Button):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        with self.canvas:
            self.icon = Image(source=self.source)
        self.bind(pos=self.update_icon, size=self.update_icon)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)

    def update_icon(self, *args):
        self.icon.pos = self.pos
        self.icon.size = self.size
