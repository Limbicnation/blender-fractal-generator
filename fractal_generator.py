bl_info = {
    "name": "Fractal Geometry Generator",
    "author": "Your Name",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Fractal",
    "description": "[WIP] Generate fractal-based geometry modifications",
    "warning": "Experimental - Use with caution",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
import bmesh
import random
import math
from bpy.props import (
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty
)

def batch_process_faces(bm, faces, operation_func, batch_size=10):
    """Process faces in smaller batches to prevent freezing"""
    for i in range(0, len(faces), batch_size):
        batch = faces[i:i + batch_size]
        for face in batch:
            operation_func(face)
        # Force update after each batch
        bmesh.update_edit_mesh(bm.mesh)

def calculate_fractal_value(x, y, max_iter=50):
    """Optimized Mandelbrot calculation with early bailout"""
    c = complex(x, y)
    z = 0
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 4:  # Optimized bailout condition
            return i / max_iter
    return 1.0

class FRACTAL_PT_main_panel(bpy.types.Panel):
    bl_label = "Fractal Generator [WIP]"
    bl_idname = "FRACTAL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Fractal'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Warning box for WIP status
        box = layout.box()
        box.label(text="⚠️ Work in Progress", icon='ERROR')
        box.label(text="Save your work before using")
        
        # Face Selection Settings
        box = layout.box()
        box.label(text="Face Selection")
        box.prop(scene, "fractal_selected_only")
        box.prop(scene, "fractal_face_limit")
        box.prop(scene, "fractal_batch_size")
        
        # Main Settings
        box = layout.box()
        box.label(text="Fractal Settings")
        box.prop(scene, "fractal_random_seed")
        box.prop(scene, "fractal_iterations")
        box.prop(scene, "fractal_scale")
        box.prop(scene, "fractal_min_depth")
        box.prop(scene, "fractal_inset_depth")
        box.prop(scene, "fractal_secondary_inset")
        box.prop(scene, "fractal_extrusion_strength")
        box.prop(scene, "fractal_recursion_chance")
        
        # Generate Button
        row = layout.row(align=True)
        row.scale_y = 2.0
        row.operator("mesh.fractal_generate", text="Generate Fractal", icon='OUTLINER_OB_SURFACE')
        
        # Cancel Button (during processing)
        if context.window_manager.fractal_is_processing:
            row = layout.row()
            row.operator("mesh.fractal_cancel", text="Cancel Generation", icon='X')

class MESH_OT_fractal_generate(bpy.types.Operator):
    bl_idname = "mesh.fractal_generate"
    bl_label = "Generate Fractal"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or context.window_manager.fractal_is_processing:
            return False
        # If selected faces only is enabled, require face selections in edit mode
        if context.scene.fractal_selected_only:
            return (obj.mode == 'EDIT' and 
                   bool(context.edit_object.data.total_face_sel))
        return True

    def modal(self, context, event):
        if event.type == 'ESC' or context.window_manager.fractal_should_cancel:
            self.cancel(context)
            return {'CANCELLED'}

        if self.current_batch < len(self.face_batches):
            self.process_batch(context, self.current_batch)
            self.current_batch += 1
            return {'RUNNING_MODAL'}
        else:
            self.finish(context)
            return {'FINISHED'}

    def process_batch(self, context, batch_idx):
        batch = self.face_batches[batch_idx]
        scene = context.scene
        
        # Ensure we're in edit mode for BMesh operations
        if context.active_object.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
            
        for face in batch:
            # Skip invalid faces
            if not face.is_valid:
                continue
                
            center = face.calc_center_median()
            fractal_value = calculate_fractal_value(
                center.x * scene.fractal_scale,
                center.y * scene.fractal_scale,
                scene.fractal_iterations
            )
            
            if fractal_value > scene.fractal_min_depth:
                # Initial inset
                inset_result = bmesh.ops.inset_region(
                    self.bm,
                    faces=[face],
                    thickness=fractal_value * scene.fractal_inset_depth,
                    depth=0,
                    use_boundary=True,
                    use_even_offset=True
                )
                
                # Get the newly created faces from inset
                new_faces = [f for f in inset_result['faces']]
                
                # For each new face, randomly apply additional operations
                for new_face in new_faces:
                    if random.random() < scene.fractal_recursion_chance:
                        # Secondary inset
                        secondary_inset = bmesh.ops.inset_region(
                            self.bm,
                            faces=[new_face],
                            thickness=fractal_value * scene.fractal_secondary_inset,
                            depth=0,
                            use_boundary=True,
                            use_even_offset=True
                        )
                        
                        # Extrude the inset faces
                        for inset_face in secondary_inset['faces']:
                            extrude_result = bmesh.ops.extrude_face_region(
                                self.bm,
                                geom=[inset_face]
                            )
                            
                            # Get vertices from extrusion
                            verts = [v for v in extrude_result["geom"] 
                                   if isinstance(v, bmesh.types.BMVert)]
                            
                            # Calculate random extrusion height
                            height = fractal_value * scene.fractal_extrusion_strength
                            height *= random.uniform(0.5, 1.0)  # Add variation
                            
                            # Translate the extruded vertices
                            bmesh.ops.translate(
                                self.bm,
                                vec=inset_face.normal * height,
                                verts=verts
                            )
        
        bmesh.update_edit_mesh(self.mesh)
        context.area.tag_redraw()

    def execute(self, context):
        try:
            obj = context.active_object
            scene = context.scene
            
            # Set random seed
            seed_value = scene.fractal_random_seed
            if seed_value == 0:  # Use system time for true random
                random.seed(None)
            else:
                random.seed(seed_value)
            
            # Initialize processing state
            context.window_manager.fractal_is_processing = True
            context.window_manager.fractal_should_cancel = False
            
            # Setup bmesh
            was_in_edit_mode = (obj.mode == 'EDIT')
            if not was_in_edit_mode:
                bpy.ops.object.mode_set(mode='EDIT')
            
            self.mesh = obj.data
            self.bm = bmesh.from_edit_mesh(self.mesh)
            self.bm.faces.ensure_lookup_table()  # Ensure face indices are valid
            
            # Get faces based on selection mode
            if scene.fractal_selected_only:
                available_faces = [f for f in self.bm.faces if f.select]
            else:
                available_faces = self.bm.faces[:]
            
            # Select faces within limit
            num_faces = len(available_faces)
            face_limit = min(scene.fractal_face_limit, num_faces)
            
            if scene.fractal_selected_only:
                # Use all selected faces up to the limit
                selected_faces = available_faces[:face_limit]
            else:
                # Randomly sample from all faces
                selected_faces = random.sample(available_faces, k=face_limit)
            
            # Create batches
            self.face_batches = [
                selected_faces[i:i + scene.fractal_batch_size]
                for i in range(0, len(selected_faces), scene.fractal_batch_size)
            ]
            self.current_batch = 0
            
            # Start modal timer
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.1, window=context.window)
            wm.modal_handler_add(self)
            
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            context.window_manager.fractal_is_processing = False
            if context.active_object.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            return {'CANCELLED'}

    def cancel(self, context):
        if hasattr(self, '_timer'):
            context.window_manager.event_timer_remove(self._timer)
        if hasattr(self, 'bm'):
            self.bm.free()
        context.window_manager.fractal_is_processing = False
        context.window_manager.fractal_should_cancel = False
        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, "Fractal generation cancelled")

    def finish(self, context):
        if hasattr(self, '_timer'):
            context.window_manager.event_timer_remove(self._timer)
        if hasattr(self, 'bm'):
            self.bm.free()
        context.window_manager.fractal_is_processing = False
        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, "Fractal generation completed")

class MESH_OT_fractal_cancel(bpy.types.Operator):
    bl_idname = "mesh.fractal_cancel"
    bl_label = "Cancel Fractal Generation"
    
    def execute(self, context):
        context.window_manager.fractal_should_cancel = True
        return {'FINISHED'}

def register_properties():
    bpy.types.WindowManager.fractal_is_processing = BoolProperty(default=False)
    bpy.types.WindowManager.fractal_should_cancel = BoolProperty(default=False)
    
    # Random seed property
    bpy.types.Scene.fractal_random_seed = IntProperty(
        name="Random Seed",
        description="Seed value for random pattern generation (0 for true random)",
        default=0,
        min=0,
        max=9999
    )
    
    # Add new property for selected faces only
    bpy.types.Scene.fractal_selected_only = BoolProperty(
        name="Selected Faces Only",
        description="Apply fractal only to selected faces",
        default=False
    )
    
    bpy.types.Scene.fractal_face_limit = IntProperty(
        name="Face Limit",
        description="Maximum number of faces to process",
        default=100,
        min=1,
        max=1000
    )
    bpy.types.Scene.fractal_batch_size = IntProperty(
        name="Batch Size",
        description="Number of faces to process per batch",
        default=10,
        min=1,
        max=50
    )
    bpy.types.Scene.fractal_iterations = IntProperty(
        name="Iterations",
        description="Number of fractal iterations",
        default=50,
        min=1,
        max=200
    )
    bpy.types.Scene.fractal_scale = FloatProperty(
        name="Scale",
        description="Scale of the fractal pattern",
        default=1.0,
        min=0.0,
        max=10.0
    )
    bpy.types.Scene.fractal_min_depth = FloatProperty(
        name="Minimum Depth",
        description="Minimum fractal value to trigger operations",
        default=0.1,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.fractal_inset_depth = FloatProperty(
        name="Initial Inset Depth",
        description="Depth of the first inset operation",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.fractal_secondary_inset = FloatProperty(
        name="Secondary Inset Depth",
        description="Depth of the secondary inset operations",
        default=0.2,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.fractal_extrusion_strength = FloatProperty(
        name="Extrusion Strength",
        description="Strength of the extrusion operations",
        default=0.5,
        min=0.0,
        max=2.0
    )
    
    bpy.types.Scene.fractal_recursion_chance = FloatProperty(
        name="Recursion Chance",
        description="Probability of additional inset/extrude operations (0-1)",
        default=0.7,
        min=0.0,
        max=1.0
    )

def unregister_properties():
    del bpy.types.WindowManager.fractal_is_processing
    del bpy.types.WindowManager.fractal_should_cancel
    del bpy.types.Scene.fractal_selected_only
    del bpy.types.Scene.fractal_random_seed
    del bpy.types.Scene.fractal_face_limit
    del bpy.types.Scene.fractal_batch_size
    del bpy.types.Scene.fractal_iterations
    del bpy.types.Scene.fractal_scale
    del bpy.types.Scene.fractal_min_depth
    del bpy.types.Scene.fractal_inset_depth
    del bpy.types.Scene.fractal_secondary_inset
    del bpy.types.Scene.fractal_extrusion_strength
    del bpy.types.Scene.fractal_recursion_chance

classes = (
    FRACTAL_PT_main_panel,
    MESH_OT_fractal_generate,
    MESH_OT_fractal_cancel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_properties()

def unregister():
    unregister_properties()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()