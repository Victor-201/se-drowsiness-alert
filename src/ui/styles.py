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
''')