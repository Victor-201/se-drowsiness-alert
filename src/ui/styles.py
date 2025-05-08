from kivy.lang import Builder

Builder.load_string('''
<ModernButton@Button>:
    background_normal: ''
    background_color: 0, 0, 0, 0
    canvas.before:
        Color:
            rgba: (0.2, 0.6, 1, 0.8) if self.state == 'normal' else (0.1, 0.5, 0.9, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [20]

<WarningButton@Button>:
    background_normal: ''
    background_color: 0, 0, 0, 0
    canvas.before:
        Color:
            rgba: (1, 0.2, 0.2, 0.8) if self.state == 'normal' else (0.9, 0.1, 0.1, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [20]

<ModernLabel@Label>:
    color: 1, 1, 1, 1
    font_size: '20sp'
    font_name: 'Roboto'
    bold: True

<MetricsPanel@BoxLayout>:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: 0.1, 0.1, 0.15, 0.9
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [25]

<StatusBar@Widget>:
    canvas:
        Color:
            rgba: 0.2, 0.2, 0.25, 1
        RoundedRectangle:
            pos: self.x + (self.width - self.bar_length) / 2, self.y + (self.height - self.bar_height) / 2
            size: self.bar_length, self.bar_height
            radius: [10]
        Color:
            rgba: self.bar_color
        RoundedRectangle:
            pos: self.x + (self.width - self.bar_length) / 2, self.y + (self.height - self.bar_height) / 2
            size: self.filled_length, self.bar_height
            radius: [10]
''')