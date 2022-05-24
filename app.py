import threading
from time import sleep
from pipeline import Pipeline
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from src import enums, viz
import os
from pathlib import Path
from vid import VideoWindow
import cv2
import matplotlib.cm as cm
# from PIL import Image


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

    def __init__(self, width, height, data_path=''):

        self.window = gui.Application.instance.create_window("COLMAP Slam", width, height)

        # Default config stuff
        self.img_count = 0
        self.pt_count = -1
        self.frame_skip = 2
        self.extractor = enums.Extractors(1)
        self.matcher = enums.Matchers(1)
        self.selector = enums.ImageSelectionMethod(1)
        self.image_path = data_path
        self.output_path = "./out/test1"
        self.export_name = "reconstruction.ply"
        self.frames = []
        self.init_frames = 20

        try:
            self.raw_img_count = len(os.listdir(data_path))
        except:
            print("Error opening path")
            self.raw_img_count = 0

        self.frame_final = self.raw_img_count

        self.rec = Pipeline()
        self.show_cam = True
        self.show_path = True
        self.show_track = -1
        self.cam_scale = 10
        self.is_setup = False
        self.start_img = 0
        self.end_img = 0
        self.last_keyframe = 0
        self.current_frame = 0
        
        # default material
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.base_color = (1, 1, 1, 1)
        self.mat.base_reflectance = 0.1

        self.mat.point_size = 10 * self.window.scaling

        w = self.window 
        self.vid = VideoWindow()
        
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Reconstruction 3d widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)


        # Create the settings panel on the right
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # File path setting

        _data_loading = gui.Horiz(0,  gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._settings_panel.add_child(gui.Label("Data Settings"))
        self._data_path = gui.Label(f"Data Path:\n {self.image_path}")
        self._settings_panel.add_child(self._data_path)

        self._out_path = gui.Label(f"Output Path:\n {self.output_path}")
        self._settings_panel.add_child(self._out_path)

        self._settings_panel.add_child(gui.Label("Edit Path"))
        _data_selector = gui.Button("Data")
        _data_selector.set_on_clicked(self._on_data_open)

        _output_selector = gui.Button("Output")
        _output_selector.set_on_clicked(self._on_out_open)

        _data_loading.add_child(_data_selector)
        _data_loading.add_child(_output_selector)

        self._settings_panel.add_child(_data_loading)
        
        self._settings_panel.add_child(gui.Label("Number of frames to skip"))
        _frame_skip = gui.Slider(gui.Slider.INT)
        _frame_skip.set_limits(1, 30)
        _frame_skip.int_value = self.frame_skip
        _frame_skip.set_on_value_changed(self._on_frame_skip)
        self._settings_panel.add_child(_frame_skip)

        self._settings_panel.add_child(gui.Label("Max number of frames"))
        self._frame_final = gui.Slider(gui.Slider.INT)
        self._frame_final.set_limits(0, self.raw_img_count // self.frame_skip)
        self._frame_final.int_value = self.raw_img_count // self.frame_skip
        self._frame_final.set_on_value_changed(self._on_frame_final)
        self._settings_panel.add_child(self._frame_final)

        self._settings_panel.add_child(gui.Label("Max frames for initialization"))
        _init_frames = gui.Slider(gui.Slider.INT)
        _init_frames.set_limits(0, 60)
        _init_frames.int_value = self.init_frames
        _init_frames.set_on_value_changed(self._on_init_frames)
        self._settings_panel.add_child(_init_frames)


        # Next basic reconstruction settings
        self._settings_panel.add_child(gui.Label("Reconstruction Settings"))


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
        _show_cam_path.checked = self.show_path
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_cam_path)

        _show_cams = gui.Checkbox("Show Cameras")
        _show_cams.set_on_checked(self._on_show_cams)
        _show_cams.checked = self.show_cam
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_show_cams)

        view_ctrls.add_child(gui.Label("Show Camera Track"))
        self._camera_tracks = gui.Slider(gui.Slider.INT)
        self._camera_tracks.set_limits(-1, self.pt_count)
        self._camera_tracks.int_value = self.show_track
        self._camera_tracks.set_on_value_changed(self._on_show_tracks)
        view_ctrls.add_child(self._camera_tracks)

        view_ctrls.add_child(gui.Label("Camera Scale"))
        _cam_size = gui.Slider(gui.Slider.DOUBLE)
        _cam_size.set_limits(0, 50)
        _cam_size.set_on_value_changed(self._on_cam_scale)
        _cam_size.double_value = self.cam_scale
        view_ctrls.add_child(_cam_size)

        view_ctrls.add_child(gui.Label("Point Scale"))
        _pt_size = gui.Slider(gui.Slider.DOUBLE)
        _pt_size.set_limits(3, 50)
        _pt_size.set_on_value_changed(self._on_pt_scale)
        _pt_size.double_value = 10
        view_ctrls.add_child(_pt_size)

        _reset_view = gui.Button("Reset Camera")
        _reset_view.set_on_clicked(self._reset_view)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(_reset_view)

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
    def _on_frame_final(self, val):
        self.frame_final = int(val)

    def _on_frame_skip(self, val):
        self.frame_skip = int(val)
        self._frame_final.int_value = self.raw_img_count // self.frame_skip
        self._frame_final.set_limits(0, self.raw_img_count // self.frame_skip)
        
    def _on_init_frames(self, val):
        self.init_frames = int(val)

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height, self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_extractor(self, name, idx):
        print(f"now using {name}")
        self.extractor = enums.Extractors(idx+1)

    def _on_matcher(self, name, idx):
        print(f"now using {name}")
        self.matcher = enums.Matchers(idx+1)

    def _on_show_path(self, show):
        if self._scene.scene.has_geometry("__path__"):
            self._scene.scene.remove_geometry("__path__")

        if show and self.start_img < self.end_img:
            # path = viz.generate_path(self.rec.reconstruction, lambda img: img > self.start_img and img < self.end_img)

            path = viz.generate_path(self.rec.reconstruction)
            print(path)
            if len(path.points) > 0:
                
                true_path = viz.generate_true_path(self.rec.image_path.parent / 'groundtruth.txt')

                scale1 = np.linalg.norm(path.get_max_bound() - path.get_min_bound())
                scale2 = np.linalg.norm(true_path.get_max_bound() - true_path.get_min_bound())

                true_path.translate(path.get_center(), False)

                # true_path.scale(scale2/scale1, path.get_center())

                if not self._scene.scene.has_geometry("__true_path__"):
                    self._scene.scene.add_geometry("__true_path__", true_path, self.mat)


                self._scene.scene.add_geometry("__path__", path, self.mat)

    def _on_show_tracks(self, pt_id):
        if self._scene.scene.has_geometry("__track__"):
            self._scene.scene.remove_geometry("__track__")

        if pt_id > 0:
            # track = viz.generate_tracks(self.rec.reconstruction, int(pt_id),lambda elem: elem.image_id > self.start_img and elem.image_id < self.end_img)
            track = viz.generate_tracks(self.rec.reconstruction, int(pt_id))

            if len(track.points) > 0:
                
                print(track)
                self._scene.scene.add_geometry("__track__", track, self.mat)

    def _on_show_cams(self, show):
        if self._scene.scene.has_geometry("__cams__"):
            self._scene.scene.remove_geometry("__cams__")

        if show and self.start_img < self.end_img:
            cams = viz.generate_cams(self.rec.reconstruction, self.cam_scale, lambda img: img > self.start_img and img < self.end_img)
            # cams = viz.generate_cams(self.rec.reconstruction, self.cam_scale)

            if len(cams.points) > 0:
                self._scene.scene.add_geometry(f"__cams__", cams, self.mat)

    def _on_cam_scale(self, val):
        self.cam_scale = val
        self._on_show_cams(self.show_cam)

    def _on_pt_scale(self, val):
        self.mat.point_size = val * self.window.scaling
        self._scene.scene.update_material(self.mat)

    def _on_start_img(self, val):
        self.start_img = val
        self._end_img.set_limits(val, self.img_count)
        self.update_pts()
        self._on_show_cams(self.show_cam)
        self._on_show_path(self.show_path)
        self._on_show_tracks(self.show_track)

    def _on_end_img(self, val):
        self.end_img = val
        self._start_img.set_limits(0,val)
        self.update_pts()
        self._on_show_cams(self.show_cam)
        self._on_show_path(self.show_path)
        self._on_show_tracks(self.show_track)

    def _on_bg_color(self, new_color):
        self._scene.scene.set_background([new_color.red, new_color.green, new_color.blue, new_color.alpha])

    def _on_point_size(self, size):
        self._point_size.double_value = int(size)

    def _reset_view(self):
        
        pt_bounds = viz.generate_pts(self.rec.reconstruction).get_axis_aligned_bounding_box()
        cam_bounds = viz.generate_cams(self.rec.reconstruction, self.cam_scale).get_oriented_bounding_box()
        self._scene.look_at(pt_bounds.get_center(), cam_bounds.get_center() + (cam_bounds.get_center() - pt_bounds.get_center())/3 , np.array([0,0,-1])@cam_bounds.R)




    def _on_data_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose root folder of image data",
                             self.window.theme)

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_data_dialog_done)
        self.window.show_dialog(dlg)

    def _on_out_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose output folder for reconstruction",
                             self.window.theme)

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_out_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_data_dialog_done(self, filename):
        self.window.close_dialog()
        self.image_path = filename
        print(filename)

        try:
            self.raw_img_count = len(os.listdir(filename))

        except:
            print("Error opening path")
            self.raw_img_count = 0

        
        self._frame_final.set_limits(0, self.raw_img_count // self.frame_skip)
        self._frame_final.int_value = self.raw_img_count // self.frame_skip


        self._data_path.text = '/'.join(filename.split('/')[-2:])

    def _on_out_dialog_done(self, filename):
        self.window.close_dialog()
        self.output_path = filename
        print(filename)
        self._out_path.text = '/'.join(filename.split('/')[-2:])

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


    def _on_about_ok(self):
        self.window.close_dialog()

    def _run_reconstruction(self):
        print("Checking settings....")
        
        self.rec.reset()
        self.is_setup = False
        self.rec.extractor = self.extractor
        self.rec.matcher = self.matcher
        self.rec.selector = self.selector

        self.rec.load_data(self.image_path, self.output_path, self.export_name, init_max_num_images=int(self.init_frames), frame_skip=int(self.frame_skip), max_frame=int(self.frame_final))

        self.frames = []
        self.imgs = []
        self.last_keyframe = 0 


        for frame in self.rec.frame_names:
            # cv2.cvtColor(self.frames[self.last_keyframe], cv2.COLOR_BGR2RGB)
            img = cv2.imread(os.path.join(self.image_path, frame))
            # img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB).copy()


            # img = Image.open(os.path.join(self.image_path, frame))

            self.frames.append(img)
            self.imgs.append(o3d.io.read_image(os.path.join(self.image_path, frame)))

        self.update_keyframe()


        threading.Thread(target=self.reconstruct).start()
        # self.reconstruct()


    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def update_pts(self):
        if self._scene.scene.has_geometry("__recon__"):
            self._scene.scene.remove_geometry(f"__recon__")
        
        pts = viz.generate_pts(self.rec.reconstruction, self.rec.image_path, lambda pt: len([e for e in pt.track.elements if e.image_id > self.start_img and e.image_id < self.end_img ]) > 0)
        self._scene.scene.add_geometry("__recon__", pts, self.mat)


    def refresh(self):

        self.img_count = max(self.rec.reconstruction.images)
        self.pt_count = len(self.rec.reconstruction.points3D)
        self.end_img  = self.img_count


        self._start_img.set_limits(0,self.img_count)
        self._end_img.set_limits(0,self.img_count)
        self._end_img.int_value = self.img_count
        self._camera_tracks.set_limits(-1, self.pt_count)

        
        self._on_show_cams(self.show_cam)
        self._on_show_path(self.show_path)
        self._on_show_tracks(self.show_track)

        self.update_pts()
        
        if not self.is_setup:
            self.is_setup = True
            
            self._reset_view()

    def update_keyframe(self):
        
        img = self.frames[self.last_keyframe]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        

        for p in self.rec.reconstruction.images[self.last_keyframe].get_valid_points2D():
            if p.point3D_id >= 18446744073709551614:  # Invalid 3d point
                continue
            img = cv2.circle(img, p.xy.astype(np.uint16), 3, 255*np.array(cm.gist_ncar(p.point3D_id/self.pt_count)[:2]))

        self.last_keyframe_img = o3d.geometry.Image(img.astype(np.uint8))

        # self.vid.kf_widget.update_image(self.last_keyframe_img)

    def update_frames(self):
        self.vid.kf_widget.update_image(self.last_keyframe_img)
        

        if self.current_frame == self.last_keyframe:
            self.update_keyframe()
            self.vid.f_widget.update_image(self.last_keyframe_img)
        else:
            self.vid.f_widget.update_image(self.imgs[self.current_frame])
            self.vid.kf_label.text = f'Last Key Frame: {self.last_keyframe}'
        self.vid.f_label.text = f'Current Frame: {self.current_frame}'


    # Callback to run when each keyframe is registered
    def process_frame(self, keyframe_id):
        print(f"Next keyframe: {keyframe_id}, rec has {len(self.rec.reconstruction.points3D)}")
        self.pt_count = len(self.rec.reconstruction.points3D)
        self._end_img.set_limits(0,self.img_count)

        if not self.vid:
            print("ERROR: no video window init")
            return 0

        if not self.last_keyframe_img:
            print('*** NO KEY FRAME SAVED')
            self.update_keyframe()
            print(self.last_keyframe_img)


        i=self.last_keyframe

        self.update_keyframe()
        self.last_keyframe = keyframe_id


        while i <= keyframe_id:
            # print(f"at frame {i} next keyframe is {keyframe_id}")
            self.current_frame = i
            
            # if i == keyframe_id:
            #     self.update_keyframe()
    
            if len(self.imgs)>i:
                gui.Application.instance.post_to_main_thread(self.vid.window, self.update_frames)
    
            sleep(self.vid.frame_delay)
            i+=1

        gui.Application.instance.post_to_main_thread(self.window, self.refresh)
        gui.Application.instance.post_to_main_thread(self.vid.window, self.update_output)



    def update_output(self):
        # do any formatting you want here
        out = self.rec.reconstruction.summary()

        self.vid.out_label.text = out

    def reconstruct(self):
        print(f"Running on data at {self.image_path} with {self.extractor.name} and {self.matcher.name}...")

        self._scene.scene.clear_geometry()

        self.rec.run(per_frame_callback=self.process_frame)

        print(self.rec.reconstruction.images.keys())
        print(self.rec.reconstruction.reg_image_ids())

        self.refresh()

            

def main():

    gui.Application.instance.initialize()

    w = AppWindow(1024, 768, "./data/rgbd_dataset_freiburg2_xyz/rgb/")

    gui.Application.instance.run()

if __name__ == "__main__":
    main()