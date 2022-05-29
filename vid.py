import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import threading
from pathlib import Path
import os


class VideoWindow:

    def __init__(self):
        self.rgb_images = []    

        self.window = gui.Application.instance.create_window(
            "Keyframes and live video", 1600, 600)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.frame_delay = .1

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Horiz(0.5 * em, gui.Margins(margin))

        default_img = (np.zeros((375,500)).astype(np.uint8))
        default_img1 = (np.ones((375,500)).astype(np.uint8))

        
        kf_panel = gui.Vert(0.5 * em, gui.Margins(margin))
        # self.kf_label = gui.Label("Keyframe image")
        # kf_panel.add_child(self.kf_label)
        self.kf_widget = gui.ImageWidget(o3d.geometry.Image(default_img))
        kf_panel.add_child(self.kf_widget)
        
        # f_panel = gui.Vert(0.5 * em, gui.Margins(margin))
        # self.f_label = gui.Label("Current Frame")
        # f_panel.add_child(self.f_label)
        # self.f_widget = gui.ImageWidget(o3d.geometry.Image(default_img))
        # f_panel.add_child(self.f_widget)

        settings_panel = gui.Vert(0.5 * em, gui.Margins(margin))
        # settings_panel.add_child(gui.Label("Frame update delay (sec)"))
        # _frame_delay = gui.Slider(gui.Slider.DOUBLE)
        # _frame_delay.set_limits(0, 5)
        # _frame_delay.double_value = self.frame_delay
        # _frame_delay.set_on_value_changed(self._on_frame_delay)
        # settings_panel.add_child(_frame_delay)

        settings_panel.add_child(gui.Label("Output:"))
        self.out_label = gui.Label("")
        settings_panel.add_child(self.out_label)

        self.panel.add_child(kf_panel)
        # self.panel.add_child(f_panel)
        self.panel.add_child(settings_panel)


        self.window.add_child(self.panel)

        self.is_done = False


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.panel.frame = gui.Rect(contentRect.x, contentRect.y, contentRect.width, contentRect.height)

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_frame_delay(self, val):
        self.frame_delay = val
