import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import src.enums as enums

'''
For WSL need to install mesaos
    
    sudo apt-get install libosmesa6-dev

and set opengl version with

    export LIBGL_ALWAYS_INDIRECT=0
    export MESA_GL_VERSION_OVERRIDE=4.5
    export MESA_GLSL_VERSION_OVERRIDE=450
    export LIBGL_ALWAYS_SOFTWARE=1

'''

class AppWindow:

    def __init__(self, width, height):

        self.window = gui.Application.instance.create_window("COLMAP Slam", width, height)

        # Default config stuff
        self.img_count = 0
        self.extractor = enums.Extractors(1)
        self.matcher = enums.Matchers(1)
        self.selector = enums.ImageSelectionMethod(1)
        self.data_path = ""


        w = self.window  # to make the code more concise

        # Reconstruction widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Create the settings panel on the right

        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._settings_panel.add_child(gui.Label("Reconstruction Settings"))

        # Start with reconstruction settings
        _extractor = gui.Combobox()
        for name, _ in enums.Extractors.__members__.items():
            _extractor.add_item(name)

        _extractor.set_on_selection_changed(self._on_extractor)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(gui.Label("Feature Extractor"))
        self._settings_panel.add_child(_extractor)

        _matcher = gui.Combobox()
        for name, _ in enums.Matchers.__members__.items():
            _matcher.add_item(name)

        _matcher.set_on_selection_changed(self._on_matcher)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(gui.Label("Feature Matcher"))
        self._settings_panel.add_child(_matcher)


        _selector = gui.Combobox()
        for name, _ in enums.ImageSelectionMethod.__members__.items():
            _selector.add_item(name)

        _selector.set_on_selection_changed(self._on_selector)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(gui.Label("Image Selector"))
        self._settings_panel.add_child(_selector)

        # Maybe add a frame rate thing? how many frames we want to process

        _run = gui.Button("Run")
        _run.set_on_clicked(self._run_reconstruction)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(_run)


        # Next add settings for the post reconstruction results

        view_ctrls = gui.CollapsableVert("Viz Settings", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))


        view_ctrls.add_child(gui.Label("Background Color"))
        _bg_color = gui.ColorEdit()
        _bg_color.color_value = gui.Color(1,1,1)
        _bg_color.set_on_value_changed(self._on_bg_color)
        view_ctrls.add_child(_bg_color)

        _show_cam_path = gui.Checkbox("Show Camera Path")
        _show_cam_path.set_on_checked(self._on_show_path)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_cam_path)

        _show_tracks = gui.Checkbox("Show Camera Tracks")
        _show_tracks.set_on_checked(self._on_show_tracks)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_tracks)

        _show_cams = gui.Checkbox("Show Cameras")
        _show_cams.set_on_checked(self._on_show_cams)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_cams)

        view_ctrls.add_child(gui.Label("Camera Scale"))
        _cam_size = gui.Slider(gui.Slider.DOUBLE)
        _cam_size.set_limits(0, 10)
        _cam_size.set_on_value_changed(self._on_cam_scale)
        view_ctrls.add_child(_cam_size)

        view_ctrls.add_fixed(separation_height)

        # Not sure if we want to keep these, should be used to view only a portion of the reconstruction/animate progress through the reconstruction

        view_ctrls.add_child(gui.Label("Start Image"))
        self._start_img = gui.Slider(gui.Slider.INT)
        self._start_img.set_limits(0, self.img_count)
        self._start_img.set_on_value_changed(self._on_start_img)
        view_ctrls.add_child(self._start_img)

        view_ctrls.add_child(gui.Label("End Image"))
        self._end_img = gui.Slider(gui.Slider.INT)
        self._end_img.set_limits(0, self.img_count)
        self._end_img.int_value = self.img_count
        self._end_img.set_on_value_changed(self._on_end_img)
        view_ctrls.add_child(self._end_img)


        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)
        
        # This is from the tutorial
        # http://www.open3d.org/docs/release/python_example/visualization/index.html#vis-gui-py
        # to render the panel on top of the scene
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

    # Whole bunch of on event listeners for setting changes above

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _on_extractor(self, name, idx):
        print(f"now using {name}")
        self.extractor = enums.Extractors(idx+1)

    def _on_selector(self, name, idx):
        print(f"now using {name}")
        self.selector = enums.ImageSelectionMethod(idx+1)


    def _on_matcher(self, name, idx):
        print(f"now using {name}")
        self.matcher = enums.Matchers(idx+1)


    def _on_show_path(self, show):
        print(show)

    def _on_show_tracks(self, show):
        print(show)

    def _on_show_cams(self, show):
        print(show)

    def _on_cam_scale(self, val):
        self.cam_scale = val
        print(val)

    def _on_start_img(self, val):
        self.start_img = val
        self._end_img.set_limits(val, self.img_count)

    def _on_end_img(self, val):
        self.end_img = val
        self.start_img.set_limit(0,val)

    def _on_bg_color(self, new_color):
        self._scene.scene.set_background([new_color.red, new_color.green, new_color.blue, new_color.alpha])

    def _on_point_size(self, size):
        self._point_size.double_value = int(size)

    # Can remove these, may want to keep the file opener for the settings...

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _run_reconstruction(self):
        print("Checking settings....")

        self.reconstruct()

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def reconstruct(self):
        print(f"Running on data at {self.data_path} with {self.extractor.name}, {self.matcher.name} and {self.selector.name}...")

def main():

    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()